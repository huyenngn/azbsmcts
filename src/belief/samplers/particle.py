from __future__ import annotations

import copy
import dataclasses
import logging
import random
from collections import abc as cabc

import numpy as np

import openspiel
from utils import utils

OpponentPolicy = cabc.Callable[[list[openspiel.State]], list[np.ndarray]]

logger = logging.getLogger(__name__)

INITIAL_STATE_SERIALIZED = "\n"


@dataclasses.dataclass
class _StepRecord:
  """Record of one real move from the AI player's perspective."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: bytes
  particles: dict[str, _Particle] | None


@dataclasses.dataclass
class _Particle:
  serialized: str
  weight: float


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    num_particles: int = 50,
    matches_per_particle: int = 15,
    rebuild_tries: int = 5,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
    temperature: float = 1.0,
    checkpoint_interval: int = 5,
  ):
    if num_particles <= 0:
      raise ValueError("num_particles must be > 0")
    if matches_per_particle <= 0:
      raise ValueError("matches_per_particle must be > 0")
    if rebuild_tries <= 0:
      raise ValueError("rebuild_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.num_particles = num_particles
    self.matches_per_particle = matches_per_particle
    self.rebuild_tries = rebuild_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy
    self.temperature = temperature
    self.checkpoint_interval = checkpoint_interval
    self._history: list[_StepRecord] = []
    self._particles: dict[str, _Particle] = {}
    self._last_valid_sample: tuple[str, float] = (
      INITIAL_STATE_SERIALIZED,
      1.0,
    )
    self._checkpoint_indices: set[int] = set()

  def sample_unique_particles(self, n: int) -> list[openspiel.State]:
    """Return up to n unique particles as deserialized game states."""
    if not self._particles:
      return [self.game.deserialize_state(self._last_valid_sample[0])]

    return [
      self.game.deserialize_state(p.serialized)
      for p in self.rng.sample(
        list(self._particles.values()),
        min(n, len(self._particles)),
      )
    ]

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particles.clear()
    self._last_valid_sample = (INITIAL_STATE_SERIALIZED, 1.0)
    self._checkpoint_indices.clear()

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a real move.

    actor/action describe the real move taken in the environment.
    real_state_after is used ONLY to extract the AI observation (non-cheating).
    """

    rec = _StepRecord(
      actor_is_ai=(actor == self.ai_id),
      ai_action=(action if actor == self.ai_id else None),
      ai_obs_after=self._ai_obs(real_state_after),
      particles=None,
    )
    self._history.append(rec)

    if rec.actor_is_ai:
      self._particles = self._resample_particles_for_ai(rec)
    else:
      self._particles = self._resample_particles_for_opponent(rec.ai_obs_after)

  def sample(self) -> str:
    """Sample a particle consistent with current observation history."""
    gamma, _ = self.sample_with_prior()
    return gamma

  def sample_with_prior(self) -> tuple[str, float]:
    """Sample a particle with prior probability."""
    if not self._history:
      return INITIAL_STATE_SERIALIZED, 1.0

    if not self._particles:
      self._rebuild_particles()

    if not self._particles:
      logger.debug(
        "Particle sampler belief collapsed with empty particles after "
        "rebuild; falling back to last valid sample."
      )
      return self._last_valid_sample

    chosen_board = self.rng.choice(list(self._particles.keys()))
    chosen_particle = self._particles[chosen_board]

    total_weight = sum(p.weight for p in self._particles.values())
    prior = chosen_particle.weight / total_weight
    self._last_valid_sample = (chosen_particle.serialized, prior)
    return chosen_particle.serialized, prior

  def _ai_obs(self, state: openspiel.State) -> bytes:
    return np.asarray(
      state.observation_tensor(self.ai_id), dtype=np.float32
    ).tobytes()

  def _get_or_create_transition(
    self, parent: str, action: int
  ) -> tuple[str, str, bytes] | None:
    """Get or compute (child_hash, child_state, observation) for a transition.

    Returns None if the action is not legal for this parent state.
    """
    s = self.game.deserialize_state(parent)
    if action not in s.legal_actions():
      return None
    s.apply_action(action)
    obs = self._ai_obs(s)
    child_hash = s.hash()
    child_str = s.serialize()
    return child_hash, child_str, obs

  def _resample_particles_for_ai(
    self,
    rec: _StepRecord,
    particles: dict[str, _Particle] | None = None,
    num_particles: int | None = None,
  ) -> dict[str, _Particle]:
    assert rec.actor_is_ai and rec.ai_action is not None
    if particles is None:
      particles = self._particles
    if num_particles is None:
      num_particles = len(particles)

    new_particles: dict[str, _Particle] = {}
    for _, particle in particles.items():
      result = self._get_or_create_transition(
        particle.serialized, rec.ai_action
      )
      if result is None:
        continue
      child_hash, child_str, obs = result
      if obs == rec.ai_obs_after:
        if child_hash not in new_particles:
          new_particles[child_hash] = _Particle(
            serialized=child_str, weight=particle.weight
          )
        else:
          new_particles[child_hash].weight += particle.weight

    return new_particles

  def _get_opponent_action_weights(
    self,
    states: list[openspiel.State],
  ) -> list[list[float]]:
    """Get action weights for states."""
    if not states:
      return []

    if self.opponent_policy is not None:
      all_policy_probs = self.opponent_policy(states)
      results: list[list[float]] = []
      for state, policy_probs in zip(states, all_policy_probs, strict=True):
        legal = state.legal_actions()
        if not legal:
          results.append([])
          continue
        probs = np.array([policy_probs[a] for a in legal], dtype=np.float32)
        probs = np.maximum(probs, 1e-12)
        results.append(probs.tolist())
      return results

    return [
      (
        np.ones(len(s.legal_actions()), dtype=np.float32)
        / len(s.legal_actions())
      ).tolist()
      for s in states
    ]

  def _resample_particles_for_opponent(
    self,
    target_obs: bytes,
    particles: dict[str, _Particle] | None = None,
    matches_per_particle: int | None = None,
    num_particles: int | None = None,
    temperature: float = 1.0,
  ) -> dict[str, _Particle]:
    if particles is None:
      particles = self._particles
    if matches_per_particle is None:
      matches_per_particle = self.matches_per_particle
    if num_particles is None:
      num_particles = self.num_particles

    particle_data: list[tuple[str, list[int], float]] = []
    deserialized_states: list[openspiel.State] = []
    for _, particle in particles.items():
      s = self.game.deserialize_state(particle.serialized)
      legal = s.legal_actions()
      if legal:
        particle_data.append((particle.serialized, legal, particle.weight))
        deserialized_states.append(s)

    if not particle_data:
      return {}

    if "go" in self.game.name:
      tmp_data = particle_data[0]
      tmp = self.game.deserialize_state(tmp_data[0])
      tmp.apply_action(tmp_data[1][-1])
      if target_obs == self._ai_obs(tmp):
        results: dict[str, _Particle] = {}
        for s, (_, legal, weight) in zip(
          deserialized_states, particle_data, strict=True
        ):
          s.apply_action(legal[-1])
          child_hash = s.hash()
          child_str = s.serialize()
          if child_hash not in results:
            results[child_hash] = _Particle(
              serialized=child_str, weight=weight
            )
          else:
            results[child_hash].weight += weight
        return results

    all_probs = self._get_opponent_action_weights(deserialized_states)

    new_particles: dict[str, _Particle] = {}
    for (parent_serialized, legal, weight), probs in zip(
      particle_data, all_probs, strict=True
    ):
      if not probs:
        continue

      if "go" in self.game.name:
        untried_indices = set(range(len(legal) - 1))
      else:
        untried_indices = set(range(len(legal)))
      match_count = 0

      while len(untried_indices) > 0:
        if match_count >= matches_per_particle:
          break

        untried = list(untried_indices)
        w = [probs[i] for i in untried]
        action_idx = self.rng.choices(untried, weights=w, k=1)[0]
        untried_indices.remove(action_idx)

        action = legal[action_idx]
        result = self._get_or_create_transition(parent_serialized, action)
        if result is None:
          continue
        child_hash, child_str, obs = result
        if obs != target_obs:
          continue

        new_weight = probs[action_idx] * weight
        if child_hash not in new_particles:
          new_particles[child_hash] = _Particle(
            serialized=child_str, weight=new_weight
          )
        else:
          new_particles[child_hash].weight += new_weight
        match_count += 1

    if not new_particles:
      return {}

    if len(new_particles) > num_particles:
      sorted_particles = sorted(
        new_particles.items(), key=lambda x: x[1].weight, reverse=True
      )
      new_particles = dict(sorted_particles[:num_particles])

    # Apply temperature to weights
    weight_array = np.array(
      [p.weight for p in new_particles.values()], dtype=np.float32
    )
    adjusted_weights = utils.apply_temp(weight_array, temperature=temperature)

    for (_, particle), new_weight in zip(
      new_particles.items(), adjusted_weights
    ):
      particle.weight = new_weight

    return new_particles

  def _rebuild_particles(self) -> None:
    """Rebuild particles using the stored observation history."""
    if not self._history:
      self._particles = {}
      return

    logger.debug(
      f"Rebuilding particles with history of {len(self._history)} steps and {len(self._particles)} existing particles."
    )

    sorted_checkpoint_indices = sorted(self._checkpoint_indices, reverse=True)
    checkpoint_step = 1 + len(sorted_checkpoint_indices) // self.rebuild_tries

    logger.debug(
      f"checkpoint indices: {sorted_checkpoint_indices}. checkpoint step size: {checkpoint_step}"
    )

    for attempt in range(self.rebuild_tries):
      if (tmp := (attempt * checkpoint_step)) < len(
        sorted_checkpoint_indices
      ) and attempt < self.rebuild_tries - 1:
        start_idx = sorted_checkpoint_indices[tmp]
        particles = self._history[start_idx].particles
      else:
        particles = {
          self.game.deserialize_state(
            INITIAL_STATE_SERIALIZED
          ).hash(): _Particle(serialized=INITIAL_STATE_SERIALIZED, weight=1.0)
        }
        start_idx = 0

      logger.debug(
        f"Rebuild attempt {attempt + 1}/{self.rebuild_tries} starting from index {start_idx}."
      )
      history_scale = (
        1.0 + (len(self._history) - start_idx) / self.game.max_game_length()
      )
      attempt_scale = 1.0 + attempt / self.rebuild_tries
      num_particles = int(self.num_particles * history_scale * attempt_scale)
      matches_per_particle = min(
        int(self.matches_per_particle * history_scale * attempt_scale),
        self.game.num_distinct_actions(),
      )
      temperature = self.temperature * history_scale * attempt_scale

      logger.debug(
        f"Rebuilding with num_particles={num_particles} "
        f"matches_per_particle={matches_per_particle} "
        f"temperature={temperature:.2f}"
      )

      ok = True
      for idx in range(start_idx, len(self._history)):
        rec = self._history[idx]
        if not particles:
          ok = False
          break

        if rec.actor_is_ai:
          particles = self._resample_particles_for_ai(
            rec, particles, num_particles=num_particles
          )
        else:
          particles = self._resample_particles_for_opponent(
            rec.ai_obs_after,
            particles,
            matches_per_particle,
            num_particles=num_particles,
            temperature=temperature,
          )

        if (
          idx != 0
          and (
            idx in self._checkpoint_indices
            or idx % self.checkpoint_interval == 0
          )
          and (not rec.particles or len(rec.particles) < len(particles))
        ):
          logger.debug(
            f"Updating checkpoint at index {idx} with {len(particles)} particles."
          )
          self._checkpoint_indices.add(idx)
          rec.particles = copy.deepcopy(particles)

      if ok and particles:
        if attempt > 0:
          checkpoint_idx = len(self._history) - 1
          self._checkpoint_indices.add(checkpoint_idx)
          rec = self._history[checkpoint_idx]
          rec.particles = copy.deepcopy(particles)
          logger.debug(
            f"Final checkpoint updated with {len(particles)} particles."
          )

        self._particles = particles
        return
