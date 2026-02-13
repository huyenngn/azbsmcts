from __future__ import annotations

import dataclasses
import random
import typing as t

import numpy as np

from belief import samplers, tree

if t.TYPE_CHECKING:
  import openspiel


class Agent(t.Protocol):
  """Protocol for game-playing agents."""

  def select_action(self, state: openspiel.State) -> int:
    """Select an action given the current game state."""
    ...


@dataclasses.dataclass
class AgentConfig:
  """Configuration for agent initialization."""

  seed: int = 0


class MCTSAgent:
  """MCTS base agent class providing RNG and utilities."""

  def __init__(
    self,
    game: openspiel.Game,
    player_id: int,
    sampler: samplers.DeterminizationSampler,
    c_uct: float,
    T: int,
    S: int,
    seed: int,
    lambda_guess: float,
    lambda_predict: float,
    length_discount: float,
  ):
    self.game = game
    self.player_id = player_id
    self.tree = tree.BeliefTree()
    self.sampler = sampler
    self.c_uct = c_uct
    self.T = T
    self.S = S
    self.lambda_guess = lambda_guess
    self.lambda_predict = lambda_predict
    self.length_discount = length_discount
    self.rng = random.Random(seed)

  def obs_key(self, state: openspiel.State, player_id: int) -> str:
    """Return observation string for the specified player."""
    return state.observation_string(player_id)

  def obs_tensor(self, state: openspiel.State, player_id: int) -> np.ndarray:
    """Return observation tensor for the specified player."""
    return np.asarray(state.observation_tensor(player_id), dtype=np.float32)

  def _terminal_value(self, state: openspiel.State) -> float:
    """Return from terminal state."""
    if "go" in self.game.name:
      return float(state.returns()[self.player_id]) * (
        self.length_discount ** state.game_length()
      )
    return float(state.returns()[self.player_id])


class PolicyTargetMixin:
  """Mixin for agents that provide policy targets for training."""

  def select_action_with_pi(
    self, state: openspiel.State
  ) -> tuple[int, np.ndarray]:
    raise NotImplementedError
