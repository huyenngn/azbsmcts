"""Belief-state tree for BS-MCTS (Wang et al., 2015).

Implements the belief-state tree data structure from
"Belief-state Monte-Carlo tree search for Phantom games."
Each node is a *belief-state* B containing per-determinization (γ) statistics.

Key equations
-------------
* Running-average update (Eq 2):
    N(γ,a) += 1
    U(γ,a) += (R - U(γ,a)) / N(γ,a)

* Belief-weighted reward (Eq 3):
    U(B,a) = Σ_i  b_i · U(γ_i, a)

* Opponent Guessing – quantal response beliefs (Eq 4-5):
    U(γ) = Σ_a U(γ,a)·N(γ,a) / Σ_a N(γ,a)
    b_i   = exp(λ·P(γ_i)·U(γ_i)) / Σ_j exp(λ·P(γ_j)·U(γ_j))

* Opponent Predicting (Eq 6):
    Pro(a_i) = exp(λ·U(B_opp,a_i)) / Σ_j exp(λ·U(B_opp,a_j))

* UCT selection on player nodes (Eq 1):
    V(B,a) = U(B,a) + c·√(ln N(B) / N(B,a))
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from utils import softmax

# ---------------------------------------------------------------------------
# Per-γ statistics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GammaActionStats:
  """Statistics for a single (γ, action) pair – Eq 2."""

  n: int = 0
  u: float = 0.0  # running average reward U(γ, a)


@dataclasses.dataclass
class GammaStats:
  """Per-determinization statistics stored inside a belief-state node."""

  n: int = 0  # N(γ) – visit count of this determinization
  prior: float = 1.0  # P(γ) – prior probability (default uniform)
  actions: dict[int, GammaActionStats] = dataclasses.field(
    default_factory=dict
  )

  # -- helpers ---------------------------------------------------------------

  def expected_utility(self) -> float:
    """U(γ) = Σ_a U(γ,a)·N(γ,a) / Σ_a N(γ,a)  (Eq 5)."""
    num = 0.0
    den = 0
    for a_stat in self.actions.values():
      num += a_stat.u * a_stat.n
      den += a_stat.n
    return num / den if den > 0 else 0.0

  def get_or_create_action(self, action: int) -> GammaActionStats:
    if action not in self.actions:
      self.actions[action] = GammaActionStats()
    return self.actions[action]


# ---------------------------------------------------------------------------
# Edge-level aggregate (kept for AZMCTS compatibility)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EdgeStats:
  """Aggregate statistics for a tree edge (action).

  These are **derived** from the per-γ stats in BS-MCTS but can also be
  written directly by AZMCTS which does not track per-γ data.
  """

  n: int = 0
  w: float = 0.0
  p: float = 0.0  # NN prior (used by AZMCTS only)

  @property
  def q(self) -> float:
    """Mean action value."""
    return 0.0 if self.n == 0 else self.w / self.n


# ---------------------------------------------------------------------------
# Belief-state node
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Node:
  """A node in the belief-state tree.

  Stores both:
  * Per-γ statistics (``gammas``) – used by BS-MCTS (Algorithm 1).
  * Aggregate edge statistics (``edges``) – used by AZMCTS for NN priors.
  """

  obs_key: str
  player_to_act: int
  is_expanded: bool = False
  legal_actions: list[int] = dataclasses.field(default_factory=list)
  n: int = 0  # N(B) total visits

  # Per-γ data  (key = serialized determinized state string)
  gammas: dict[str, GammaStats] = dataclasses.field(default_factory=dict)

  # Aggregate edges (kept for AZMCTS compatibility)
  edges: dict[int, EdgeStats] = dataclasses.field(default_factory=dict)

  # -- per-γ helpers --------------------------------------------------------

  def get_or_create_gamma(self, gamma_key: str) -> GammaStats:
    """Get existing γ stats or create a new entry."""
    if gamma_key not in self.gammas:
      self.gammas[gamma_key] = GammaStats()
    return self.gammas[gamma_key]

  # -- belief computation (Opponent Guessing, Eq 3-5) -----------------------

  def belief_weights(self, lambda_guess: float = 1.0) -> dict[str, float]:
    """Compute belief weights b_i for all γ ∈ B  (Eq 4).

    b_i = exp(λ · P(γ_i) · U(γ_i)) / Σ_j exp(λ · P(γ_j) · U(γ_j))

    Returns dict mapping gamma_key → b_i.
    """
    if not self.gammas:
      return {}

    keys = list(self.gammas.keys())
    logits = np.array(
      [
        lambda_guess * self.gammas[k].prior * self.gammas[k].expected_utility()
        for k in keys
      ],
      dtype=np.float64,
    )
    weights = softmax.softmax_np(logits)
    return dict(zip(keys, weights.tolist(), strict=True))

  def belief_weighted_u(self, action: int, lambda_guess: float = 1.0) -> float:
    """U(B, a) = Σ_i  b_i · U(γ_i, a)   (Eq 3).

    Only γ that have stats for *action* contribute.
    """
    bw = self.belief_weights(lambda_guess)
    if not bw:
      return 0.0

    total = 0.0
    for gk, bi in bw.items():
      gs = self.gammas[gk]
      if action in gs.actions:
        total += bi * gs.actions[action].u
    return total

  # -- aggregate visit counts -----------------------------------------------

  def total_action_visits(self, action: int) -> int:
    """N(B, a) = Σ_{γ ∈ B} N(γ, a)."""
    total = 0
    for gs in self.gammas.values():
      if action in gs.actions:
        total += gs.actions[action].n
    return total

  def total_visits(self) -> int:
    """N(B) = Σ_{γ ∈ B} N(γ)."""
    return sum(gs.n for gs in self.gammas.values())

  # -- selection helpers -----------------------------------------------------

  def uct_value(
    self, action: int, c_uct: float, lambda_guess: float = 1.0
  ) -> float:
    """V(B, a) = U(B, a) + c · √(ln N(B) / N(B, a))  (Eq 1)."""
    nb = self.total_visits()
    nba = self.total_action_visits(action)
    u_ba = self.belief_weighted_u(action, lambda_guess)
    exploration = c_uct * math.sqrt(math.log(nb + 1.0) / (nba + 1.0))
    return u_ba + exploration

  def opponent_action_probs(
    self, actions: list[int], lambda_predict: float = 1.0
  ) -> list[float]:
    """Pro(a_i) via Opponent Predicting  (Eq 6).

    Pro(a_i) = exp(λ · U(B_opp, a_i)) / Σ_j exp(λ · U(B_opp, a_j))

    Uses the aggregate belief-weighted U(B, a) for each action.
    """
    if not actions:
      return []
    logits = np.array(
      [lambda_predict * self.belief_weighted_u(a) for a in actions],
      dtype=np.float64,
    )
    return softmax.softmax_np(logits).tolist()

  # -- legacy helpers (AZMCTS compat) ------------------------------------

  def get_most_visited_action(self, actions: list[int] | None = None) -> int:
    """Return action with highest visit count (aggregate edges)."""
    if actions is None:
      actions = list(self.edges.keys())
    return max(actions, key=lambda a: self.edges.get(a, EdgeStats()).n)


# ---------------------------------------------------------------------------
# Belief tree
# ---------------------------------------------------------------------------


class BeliefTree:
  """Tree structure mapping observations to belief-state nodes."""

  def __init__(self) -> None:
    self._nodes: dict[str, Node] = {}

  def get_or_create(self, obs_key: str, player_to_act: int) -> Node:
    """Get existing node or create new one for observation."""
    if obs_key not in self._nodes:
      self._nodes[obs_key] = Node(obs_key=obs_key, player_to_act=player_to_act)
    return self._nodes[obs_key]

  def clear(self) -> None:
    """Remove all nodes."""
    self._nodes.clear()
