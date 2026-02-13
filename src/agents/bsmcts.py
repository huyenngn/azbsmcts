"""Belief-State Monte Carlo Tree Search  (Algorithm 1 from Wang et al., 2015).

Faithfully implements the BS-MCTS algorithm described in
"Belief-state Monte-Carlo tree search for Phantom games,"
including:

* Per-determinization (γ) expansion and backpropagation  (Eq 2)
* Belief-weighted reward aggregation  U(B,a)              (Eq 3)
* Opponent Guessing via quantal-response beliefs           (Eq 4-5)
* UCT selection on player nodes                            (Eq 1)
* Opponent Predicting with roulette-wheel selection        (Eq 6)
* Outer loop: T samplings × S searches per sampling
* Action selection:  argmax_a  U(B_root, a)
"""

from __future__ import annotations

import typing as t

import agents

if t.TYPE_CHECKING:
  import openspiel
  from belief import samplers, tree


class BSMCTSAgent(agents.MCTSAgent):
  """Belief-State Monte Carlo Tree Search agent  (Algorithm 1).

  Parameters
  ----------
  game : openspiel.Game
  player_id : int
      The player this agent controls (= "root player" in the paper).
  sampler : samplers.DeterminizationSampler
      Produces serialized determinized states γ.
  c_uct : float
      Exploration constant *c* in Eq 1.
  T : int
      Maximum number of samplings (outer loop).
  S : int
      Maximum number of search iterations per sampling (inner loop).
  seed : int
  lambda_guess : float
      λ for Opponent Guessing (Eq 4).  Higher → sharper beliefs.
  lambda_predict : float
      λ for Opponent Predicting (Eq 6).  Higher → opponent less
      likely to pick its best move.
  length_discount : float
      Per-ply discount applied to terminal returns to discourage stalling.
  """

  def __init__(
    self,
    game: openspiel.Game,
    player_id: int,
    sampler: samplers.DeterminizationSampler,
    c_uct: float = 1.4,
    T: int = 64,
    S: int = 8,
    seed: int = 0,
    lambda_guess: float = 1.0,
    lambda_predict: float = 1.0,
    length_discount: float = 0.999,
  ):
    super().__init__(
      game=game,
      player_id=player_id,
      sampler=sampler,
      c_uct=c_uct,
      T=T,
      S=S,
      seed=seed,
      lambda_guess=lambda_guess,
      lambda_predict=lambda_predict,
      length_discount=length_discount,
    )

  # ── public API (Algorithm 1, lines 1-13) ─────────────────────────────────

  def select_action(self, state: openspiel.State) -> int:
    """BS-MCTS main loop.

    ::

        t ← 1
        repeat
            γ ← Sampling(B_root)
            s ← 1
            repeat
                R ← Search(γ, B_root)
                N(γ) ← N(γ) + 1
                s ← s + 1
            until s > S
            t ← t + 1
        until t > T
        return argmax_a U(B_root, a)
    """
    root_obs = self.obs_key(state, self.player_id)
    root = self.tree.get_or_create(root_obs, state.current_player())

    for _ in range(self.T):
      # -- Sampling stage (Alg 1, line 4) -----------------------------------
      gamma_str, gamma_prior = self.sampler.sample_with_prior()

      # Register this γ in the root belief-state (Alg 1, lines 27-28)
      gs = root.get_or_create_gamma(gamma_str)
      gs.prior = gamma_prior

      for _ in range(self.S):
        # -- Search stage (Alg 1, line 7) ----------------------------------
        gamma_state = self.game.deserialize_state(gamma_str)
        self._search(gamma_str, gamma_state, root)

    # -- Return best action (Alg 1, line 13) --------------------------------
    # argmax_a U(B_root, a)  over legal actions
    legal = state.legal_actions()
    return max(
      legal,
      key=lambda a: root.belief_weighted_u(a, self.lambda_guess),
    )

  # ── Search (Algorithm 1, lines 30-43) ────────────────────────────────────

  def _search(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> float:
    """Recursive search on a single determinization γ through belief node B.

    ::

        if N(B) = 0 then R ← Simulation(γ); return R
        if γ has no children then Expansion(γ, B)
        N(γ) ← N(γ) + 1
        a ← Selection(γ, B)
        R ← Search(γ·a, B·a)
        N(γ,a) ← N(γ,a) + 1
        U(γ,a) ← U(γ,a) + (R − U(γ,a)) / N(γ,a)
        return R
    """
    # Terminal check
    if gamma_state.is_terminal():
      return self._terminal_value(gamma_state)

    gs = node.get_or_create_gamma(gamma_key)

    # "if N(B) = 0" – first time ANY γ visits this node → simulate
    if node.total_visits() == 0:
      gs.n += 1
      return self._rollout(gamma_state.clone())

    # "if γ has no children" – first time THIS γ visits this node → expand
    if not gs.actions:
      self._expand(gamma_key, gamma_state, node)

    gs.n += 1

    # -- Selection (Alg 1, line 39) -----------------------------------------
    action = self._selection(gamma_state, node)
    if action is None:
      # No legal action found – fall back to rollout
      return self._rollout(gamma_state.clone())

    # Descend: γ·a  and  B·a
    gamma_state.apply_action(action)
    child_obs = self.obs_key(gamma_state, self.player_id)
    child_node = self.tree.get_or_create(
      child_obs, gamma_state.current_player()
    )
    child_gamma_key = gamma_state.serialize()
    child_node.get_or_create_gamma(child_gamma_key)

    # Recurse (Alg 1, line 40)
    reward = self._search(child_gamma_key, gamma_state, child_node)

    # -- Backpropagation (Alg 1, lines 41-42) ──────────────────────────────
    a_stat = gs.get_or_create_action(action)
    a_stat.n += 1
    a_stat.u += (reward - a_stat.u) / a_stat.n  # running average (Eq 2)

    return reward

  # ── Expansion (Algorithm 1, lines 15-24) ─────────────────────────────────

  def _expand(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> None:
    """Expand γ inside belief-state B.

    ::

        N(γ) ← 0
        for all a ∈ A(γ) do
            if B·a not in tree then add B·a
            add γ·a to B·a
            N(γ,a) ← 0;  U(γ,a) ← 0
    """
    node.is_expanded = True
    legal = list(gamma_state.legal_actions())
    # Update the node's union of legal actions
    for a in legal:
      if a not in node.legal_actions:
        node.legal_actions.append(a)

    gs = node.get_or_create_gamma(gamma_key)
    for a in legal:
      gs.get_or_create_action(a)  # N(γ,a) = 0, U(γ,a) = 0

  # ── Selection (Algorithm 1, lines 45-52) ─────────────────────────────────

  def _selection(
    self,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> int | None:
    """Select action depending on whether B is a player node or opponent node.

    ::

        if B is Player node then
            a ← argmax_{a ∈ A(γ)}  V_playerNode(B, a)   [Eq 1]
        else
            a ← RouletteWheelSelection(Pro(a_i))         [Eq 6]
    """
    legal = list(gamma_state.legal_actions())
    if not legal:
      return None

    if node.player_to_act == self.player_id:
      # Player node → UCT selection (Eq 1)
      return max(
        legal,
        key=lambda a: node.uct_value(a, self.c_uct, self.lambda_guess),
      )
    # Opponent node → Roulette Wheel Selection via Opponent Predicting (Eq 6)
    probs = node.opponent_action_probs(legal, self.lambda_predict)
    return self.rng.choices(legal, weights=probs, k=1)[0]

  # ── Simulation (rollout) ─────────────────────────────────────────────────

  def _rollout(self, state: openspiel.State) -> float:
    """Random playout to terminal."""
    while not state.is_terminal():
      la = state.legal_actions()
      if not la:
        break
      state.apply_action(self.rng.choice(la))

    if state.is_terminal():
      return self._terminal_value(state)
    return 0.0

  def _terminal_value(self, state: openspiel.State) -> float:
    """Return from terminal state with length discount."""
    if self.game.name == "phantom_go":
      return float(state.returns()[self.player_id]) * (
        self.length_discount ** state.game_length()
      )
    return float(state.returns()[self.player_id])
