from __future__ import annotations

import random
from typing import Protocol

import pyspiel


class DeterminizationSampler(Protocol):
    """
    Returns a determinized state gamma suitable for simulation.
    In imperfect-information games, determinization MUST NOT be done by cloning
    the real environment state (cheating). Use a belief sampler (particles).
    """

    def sample(
        self, state: pyspiel.State, rng: random.Random
    ) -> pyspiel.State: ...
