from __future__ import annotations

import random
from dataclasses import dataclass

import pyspiel

from utils import utils


@dataclass
class _StepRecord:
    """Record of a single game step for belief reconstruction."""

    actor_is_ai: bool
    ai_action: int | None
    ai_obs_after: str


class ParticleBeliefSampler:
    """
    Non-cheating belief sampler using particle filtering.

    Maintains particles (fully-determined states) consistent with
    the AI player's observation history without using hidden information.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        ai_id: int,
        num_particles: int = 32,
        opp_tries_per_particle: int = 8,
        rebuild_max_tries: int = 200,
        seed: int = 0,
    ):
        self.game = game
        self.ai_id = ai_id
        self.num_particles = num_particles
        self.opp_tries_per_particle = opp_tries_per_particle
        self.rebuild_max_tries = rebuild_max_tries
        self.rng = random.Random(seed)

        self._history: list[_StepRecord] = []
        self._particles: list[pyspiel.State] = []

    def _ai_obs(self, state: pyspiel.State) -> str:
        return state.observation_string(self.ai_id)

    def reset(self) -> None:
        """Clear history and particles for a new game."""
        self._history.clear()
        self._particles.clear()

    def step(
        self, actor: int, action: int, real_state_after: pyspiel.State
    ) -> None:
        """
        Update belief after a move.

        Args:
            actor: Player who made the move.
            action: Action taken (ignored if actor is opponent).
            real_state_after: State after action for observation extraction.
        """
        rec = _StepRecord(
            actor_is_ai=(actor == self.ai_id),
            ai_action=(action if actor == self.ai_id else None),
            ai_obs_after=self._ai_obs(real_state_after),
        )
        self._history.append(rec)

        if not self._particles:
            self._rebuild_particles()
            return

        updated: list[pyspiel.State] = []
        for p in self._particles:
            p2 = utils.clone_state(p)

            if rec.actor_is_ai:
                if rec.ai_action not in p2.legal_actions():
                    continue
                p2.apply_action(rec.ai_action)
                if self._ai_obs(p2) == rec.ai_obs_after:
                    updated.append(p2)
                continue

            # Opponent action hidden: sample until observation matches
            ok = False
            for _ in range(self.opp_tries_per_particle):
                p3 = utils.clone_state(p2)
                la = p3.legal_actions()
                if not la:
                    break
                a = self.rng.choice(la)
                p3.apply_action(a)
                if self._ai_obs(p3) == rec.ai_obs_after:
                    updated.append(p3)
                    ok = True
                    break
            if not ok:
                pass

        self._particles = updated
        if not self._particles:
            self._rebuild_particles()

    def sample(self) -> pyspiel.State | None:
        """Sample a particle consistent with observation history."""
        if not self._particles:
            return None

        return utils.clone_state(self.rng.choice(self._particles))

    def _rebuild_particles(self) -> None:
        """Rejection sampling from initial state consistent with observation history."""
        self._particles = []
        tries = 0

        while (
            len(self._particles) < self.num_particles
            and tries < self.rebuild_max_tries
        ):
            tries += 1
            s = self.game.new_initial_state()
            ok = True

            for rec in self._history:
                if rec.actor_is_ai:
                    if rec.ai_action not in s.legal_actions():
                        ok = False
                        break
                    s.apply_action(rec.ai_action)
                    if self._ai_obs(s) != rec.ai_obs_after:
                        ok = False
                        break
                else:
                    matched = False
                    for _ in range(self.opp_tries_per_particle):
                        s2 = utils.clone_state(s)
                        la = s2.legal_actions()
                        if not la:
                            break
                        a = self.rng.choice(la)
                        s2.apply_action(a)
                        if self._ai_obs(s2) == rec.ai_obs_after:
                            s = s2
                            matched = True
                            break
                    if not matched:
                        ok = False
                        break

            if ok:
                self._particles.append(s)


class ParticleDeterminizationSampler:
    """
    Adapter conforming ParticleBeliefSampler to DeterminizationSampler.

    Falls back to cloning current state if no particles available.
    """

    def __init__(self, particle_sampler: ParticleBeliefSampler):
        self.particle_sampler = particle_sampler

    def sample(
        self, state: pyspiel.State, rng: random.Random
    ) -> pyspiel.State:
        """Sample a determinized state from particles or clone."""
        p = self.particle_sampler.sample()
        return p if p is not None else utils.clone_state(state)
