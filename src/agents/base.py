from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
import pyspiel


class Agent(Protocol):
    def select_action(self, state: pyspiel.State) -> int: ...


@dataclass
class AgentConfig:
    seed: int = 0


class BaseAgent:
    """
    Shared base:
    - RNG
    - strict observation access with explicit player id (no ambiguous defaults)
    """

    def __init__(self, player_id: int, num_actions: int, seed: int = 0):
        self.player_id = player_id
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def obs_key(self, state: pyspiel.State, player_id: int) -> str:
        # NEVER call observation_string() without player id.
        return state.observation_string(player_id)

    def obs_tensor(self, state: pyspiel.State, player_id: int) -> np.ndarray:
        # NEVER call observation_tensor() without player id.
        obs = state.observation_tensor(player_id)  # will error if unsupported
        return np.asarray(obs, dtype=np.float32).reshape(-1)


class PolicyTargetMixin:
    def select_action_with_pi(
        self, state: pyspiel.State, temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        raise NotImplementedError
