from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import pyspiel


@dataclass
class _StepRecord:
    actor_is_ai: bool
    ai_action: Optional[int]
    ai_obs_after: str


class ParticleBeliefSampler:
    """
    Non-cheating belief sampler for Phantom Go.

    Maintains K particles (fully-determined states) consistent with the AI player's
    observation history, but NEVER conditions on hidden opponent actions.
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

        self._history: List[_StepRecord] = []
        self._particles: List[pyspiel.State] = []

    def _ai_obs(self, state: pyspiel.State) -> str:
        return state.observation_string(self.ai_id)

    def reset(self):
        self._history.clear()
        self._particles.clear()

    def step(self, actor: int, action: int, real_state_after: pyspiel.State):
        """
        Call this after EVERY real ply (human or AI) was applied to the real state.
        If actor != ai_id, the action is hidden and will NOT be used (non-cheating).
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

        updated: List[pyspiel.State] = []
        for p in self._particles:
            p2 = p.clone()

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
                p3 = p2.clone()
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

    def sample(self) -> Optional[pyspiel.State]:
        """
        Returns a sampled determinization gamma as a clone of a particle.
        Returns None if no particles exist (caller must handle).
        """
        if not self._particles:
            return None
        return self.rng.choice(self._particles).clone()

    def _rebuild_particles(self):
        """
        Rejection sampling from initial state using *only* AI info:
        - AI actions (known)
        - AI observation after each ply (known)
        - opponent actions sampled (hidden)
        """
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
                        s2 = s.clone()
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
    Adapter: conform ParticleBeliefSampler to DeterminizationSampler interface.
    Degrades gracefully to cloning if particles are empty (still non-cheating).
    """

    def __init__(self, particle_sampler: ParticleBeliefSampler):
        self.particle_sampler = particle_sampler

    def sample(
        self, state: pyspiel.State, rng: random.Random
    ) -> pyspiel.State:
        p = self.particle_sampler.sample()
        return p if p is not None else state.clone()
