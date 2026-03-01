"""Belief-State Monte Carlo Tree Search (BS-MCTS) agent implementation."""

from __future__ import annotations

import typing as t

import numpy as np

from agents import base

if t.TYPE_CHECKING:
  import openspiel
  from belief import samplers, tree


class BSMCTSAgent(base.MCTSAgent):
  """Belief-State Monte Carlo Tree Search agent"""

  def __init__(
    self,
    game: openspiel.Game,
    player_id: int,
    sampler: samplers.DeterminizationSampler,
    c_uct: float = 1.5,
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

  def select_action(self, state: openspiel.State) -> int:
    root_obs = self.obs_key(state, self.player_id)
    root = self.tree.get_or_create(root_obs, state.current_player())

    for _ in range(self.T):
      gamma_str, gamma_prior = self.sampler.sample_with_prior()

      gs = root.get_or_create_gamma(gamma_str)
      gs.prior = gamma_prior

      for _ in range(self.S):
        gamma_state = self.game.deserialize_state(gamma_str)
        self._search(gamma_str, gamma_state, root)

    legal = state.legal_actions()
    return max(
      legal,
      key=lambda a: root.belief_weighted_u(a, self.lambda_guess),
    )

  def _search(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> float:
    """Recursive search procedure for belief-state MCTS."""
    if gamma_state.is_terminal():
      return self._terminal_value(gamma_state)

    gs = node.get_or_create_gamma(gamma_key)

    if node.total_visits() == 0:
      gs.n += 1
      return self._rollout(gamma_state.clone())

    if not gs.actions:
      self._expand(gamma_key, gamma_state, node)

    gs.n += 1

    action = self._selection(gamma_state, node)
    if action is None:
      return self._rollout(gamma_state.clone())

    gamma_state.apply_action(action)
    child_obs = self.obs_key(gamma_state, self.player_id)
    child_node = self.tree.get_or_create(
      child_obs, gamma_state.current_player()
    )
    child_gamma_key = gamma_state.serialize()
    child_node.get_or_create_gamma(child_gamma_key)

    reward = self._search(child_gamma_key, gamma_state, child_node)

    a_stat = gs.get_or_create_action(action)
    a_stat.n += 1
    a_stat.u += (reward - a_stat.u) / a_stat.n

    return reward

  def _expand(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> None:
    """Expand the node by adding legal actions and their corresponding child gammas."""
    node.is_expanded = True
    legal = list(gamma_state.legal_actions())
    # Update the node's union of legal actions
    for a in legal:
      if a not in node.legal_actions:
        node.legal_actions.append(a)

    gs = node.get_or_create_gamma(gamma_key)
    for a in legal:
      gs.get_or_create_action(a)

  def _selection(
    self,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> int | None:
    legal = list(gamma_state.legal_actions())
    if not legal:
      return None

    if node.player_to_act == self.player_id:
      return max(
        legal,
        key=lambda a: node.uct_value(a, self.c_uct, self.lambda_guess),
      )
    probs = np.array(
      node.opponent_action_probs(legal, self.lambda_predict), dtype=np.float64
    )
    probs = np.where(np.isfinite(probs) & (probs > 0.0), probs, 0.0)
    total = float(np.sum(probs))
    if total <= 0.0:
      probs = np.ones(len(legal), dtype=np.float64) / len(legal)
    else:
      probs /= total
    return self.rng.choices(legal, weights=probs.tolist(), k=1)[0]

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
