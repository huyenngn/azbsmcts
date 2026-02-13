"""Belief-state tree for BS-MCTS (Wang et al., 2015)."""

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
  """Statistics for a single (γ, action) pair.

  Stores both:
  * Running-average reward ``u`` for BS-MCTS (Eq 2).
  * Sum-of-values ``w`` for PUCT-based backprop (AZBSMCTS).
  * NN prior ``p`` for PUCT selection (AZBSMCTS).
  """

  n: int = 0
  u: float = 0.0  # running average reward U(γ, a)  — BS-MCTS
  w: float = 0.0  # cumulative value W(γ, a)         — AZBSMCTS
  p: float = 0.0  # NN prior P(a|γ)                  — AZBSMCTS

  @property
  def q(self) -> float:
    """Mean action value Q(γ, a) = W / N."""
    return 0.0 if self.n == 0 else self.w / self.n


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
# Belief-state node
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Node:
  """A node in the belief-state tree.

  Stores per-γ statistics (``gammas``) used by both BS-MCTS and AZBSMCTS.
  BS-MCTS uses running-average ``u`` and UCT; AZBSMCTS uses sum-based ``w``
  with NN priors ``p`` and PUCT.
  """

  obs_key: str
  player_to_act: int
  is_expanded: bool = False
  legal_actions: list[int] = dataclasses.field(default_factory=list)

  # Per-γ data  (key = serialized determinized state string)
  gammas: dict[str, GammaStats] = dataclasses.field(default_factory=dict)

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

  # -- aggregate action value (sum-based, for PUCT) -------------------------

  def aggregate_q(self, action: int) -> float:
    """Q(B, a) = Σ_γ W(γ,a) / Σ_γ N(γ,a)  — sum-based mean value."""
    total_w = 0.0
    total_n = 0
    for gs in self.gammas.values():
      if action in gs.actions:
        total_w += gs.actions[action].w
        total_n += gs.actions[action].n
    return 0.0 if total_n == 0 else total_w / total_n

  def aggregate_prior(self, action: int) -> float:
    """Weighted average NN prior P(a) across determinizations.

    P(a) = Σ_γ N(γ) · p(γ,a) / Σ_γ N(γ)   for γ that have *action*.
    Falls back to simple average if no visits yet.
    """
    total_p = 0.0
    total_n = 0
    for gs in self.gammas.values():
      if action in gs.actions:
        total_p += gs.n * gs.actions[action].p
        total_n += gs.n
    if total_n == 0:
      # No visits — return simple average of available priors
      priors = [
        gs.actions[action].p
        for gs in self.gammas.values()
        if action in gs.actions
      ]
      return sum(priors) / len(priors) if priors else 0.0
    return total_p / total_n

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

  def puct_value(self, action: int, c_puct: float) -> float:
    """PUCT score: Q(B,a) + c_puct · P(a) · √(N(B)+1) / (1 + N(B,a))."""
    nb = self.total_visits()
    nba = self.total_action_visits(action)
    q = self.aggregate_q(action)
    p = self.aggregate_prior(action)
    exploration = c_puct * p * math.sqrt(nb + 1.0) / (1.0 + nba)
    return q + exploration

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

  # -- visit-count helpers ---------------------------------------------------

  def get_most_visited_action(self, actions: list[int] | None = None) -> int:
    """Return action with highest aggregate visit count N(B, a).

    Uses per-γ stats: N(B, a) = Σ_γ N(γ, a).
    """
    if actions is None:
      actions = self.legal_actions
    if not actions:
      raise ValueError("No actions available")
    return max(actions, key=lambda a: self.total_action_visits(a))


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
