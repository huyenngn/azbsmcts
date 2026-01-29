from __future__ import annotations

import random
from typing import Protocol

import pyspiel


class DeterminizationSampler(Protocol):
    """Protocol for samplers that generate determinizations from belief states."""

    def sample(
        self, state: pyspiel.State, rng: random.Random
    ) -> pyspiel.State: ...
