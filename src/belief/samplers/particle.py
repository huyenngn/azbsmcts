from __future__ import annotations

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
  particle_weights: dict[str, float] | None


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    max_num_particles: int = 150,
    max_matches_per_particle: int = 100,
    checkpoint_interval: int = 5,
    rebuild_tries: int = 5,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
    temperature: float = 1.0,
  ):
    if max_num_particles <= 0:
      raise ValueError("max_num_particles must be > 0")
    if max_matches_per_particle <= 0:
      raise ValueError("max_matches_per_particle must be > 0")
    if checkpoint_interval <= 0:
      raise ValueError("checkpoint_interval must be > 0")
    if rebuild_tries <= 0:
      raise ValueError("rebuild_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.max_num_particles = max_num_particles
    self.max_matches_per_particle = max_matches_per_particle
    self.checkpoint_interval = checkpoint_interval
    self.rebuild_tries = rebuild_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy
    self.temperature = temperature
    self._history: list[_StepRecord] = []
    self._particle_weights: dict[str, float] = {}
    self._last_valid_sample: tuple[str, float] = (
      INITIAL_STATE_SERIALIZED,
      1.0,
    )
    self._checkpoint_indices: set[int] = set()

  def sample_unique_particles(self, n: int) -> list[openspiel.State]:
    """Return up to n unique particles as deserialized game states."""
    if not self._particle_weights:
      return []

    return [
      self.game.deserialize_state(p)
      for p in self.rng.sample(
        list(self._particle_weights.keys()),
        min(n, len(self._particle_weights)),
      )
    ]

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particle_weights.clear()
    self._last_valid_sample = (INITIAL_STATE_SERIALIZED, 1.0)
    self._checkpoint_indices.clear()

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a real move.

    actor/action describe the real move taken in the environment.
    real_state_after is used ONLY to extract the AI observation (non-cheating).
    """
    particle_weights = None
    if self._particle_weights and (
      len(self._history) % self.checkpoint_interval == 0
    ):
      particle_weights = self._particle_weights.copy()
      self._checkpoint_indices.add(len(self._history))

    rec = _StepRecord(
      actor_is_ai=(actor == self.ai_id),
      ai_action=(action if actor == self.ai_id else None),
      ai_obs_after=self._ai_obs(real_state_after),
      particle_weights=particle_weights,
    )
    self._history.append(rec)

    if rec.actor_is_ai:
      self._particle_weights = self._resample_particles_for_ai(rec)
    else:
      self._particle_weights = self._resample_particles_for_opponent(
        rec.ai_obs_after
      )

  def sample(self) -> str:
    """Sample a particle consistent with current observation history."""
    gamma, _ = self.sample_with_prior()
    return gamma

  def sample_with_prior(self) -> tuple[str, float]:
    """Sample a particle with prior probability."""
    if not self._history:
      return INITIAL_STATE_SERIALIZED, 1.0

    if not self._particle_weights:
      self._rebuild_particles()

    if not self._particle_weights:
      logger.warning(
        "Particle sampler belief collapsed with empty particles after "
        "rebuild; falling back to last valid sample."
      )
      return self._last_valid_sample

    chosen = self.rng.choice(list(self._particle_weights.keys()))

    prior = self._particle_weights[chosen] / sum(
      self._particle_weights.values()
    )
    self._last_valid_sample = (chosen, prior)
    return chosen, prior

  def _ai_obs(self, state: openspiel.State) -> bytes:
    return np.asarray(
      state.observation_tensor(self.ai_id), dtype=np.float32
    ).tobytes()

  def _get_or_create_transition(
    self, parent: str, action: int
  ) -> tuple[str, bytes] | None:
    """Get or compute (child_state, observation) for a transition.

    Returns None if the action is not legal for this parent state.
    """
    s = self.game.deserialize_state(parent)
    if action not in s.legal_actions():
      return None
    s.apply_action(action)
    obs = self._ai_obs(s)
    child_str = s.serialize()
    return child_str, obs

  def _resample_particles_for_ai(
    self,
    rec: _StepRecord,
    particle_weights: dict[str, float] | None = None,
    num_particles: int | None = None,
  ) -> dict[str, float]:
    assert rec.actor_is_ai and rec.ai_action is not None
    if particle_weights is None:
      particle_weights = self._particle_weights
    if num_particles is None:
      num_particles = len(particle_weights)

    surviving: list[tuple[str, str]] = []
    for p in set(particle_weights.keys()):
      result = self._get_or_create_transition(p, rec.ai_action)
      if result is None:
        continue
      child_str, obs = result
      if obs == rec.ai_obs_after:
        surviving.append((p, child_str))

    if surviving:
      return {child: particle_weights[p] for p, child in surviving}
    return {}

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
    particle_weights: dict[str, float] | None = None,
    matches_per_particle: int | None = None,
    num_particles: int | None = None,
    temperature: float = 1.0,
  ) -> dict[str, float]:
    if particle_weights is None:
      particle_weights = self._particle_weights
    if matches_per_particle is None:
      matches_per_particle = self.game.num_distinct_actions()
    if num_particles is None:
      num_particles = self.max_num_particles

    particle_data: list[tuple[str, list[int], float]] = []
    deserialized_states: list[openspiel.State] = []
    for particle, weight in particle_weights.items():
      s = self.game.deserialize_state(particle)
      legal = s.legal_actions()
      if legal:
        particle_data.append((particle, legal, weight))
        deserialized_states.append(s)

    if not particle_data:
      return {}

    if "go" in self.game.name:
      tmp_data = particle_data[0]
      tmp = self.game.deserialize_state(tmp_data[0])
      tmp.apply_action(tmp_data[1][-1])
      if target_obs == self._ai_obs(tmp):
        results: dict[str, float] = {}
        for s, (_, legal, weight) in zip(
          deserialized_states, particle_data, strict=True
        ):
          s.apply_action(legal[-1])
          results[s.serialize()] = weight
        return results

    all_probs = self._get_opponent_action_weights(deserialized_states)

    new_particle_weights: dict[str, float] = {}
    for (particle, legal, weight), probs in zip(
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
        result = self._get_or_create_transition(particle, action)
        if result is None:
          continue
        child_str, obs = result
        if obs != target_obs:
          continue

        new_weight = probs[action_idx] * weight
        new_particle_weights[child_str] = (
          new_particle_weights.get(child_str, 0.0) + new_weight
        )
        match_count += 1

    if not new_particle_weights:
      return {}

    if len(new_particle_weights) > num_particles:
      sorted_particles = sorted(
        new_particle_weights.items(), key=lambda x: x[1], reverse=True
      )
      new_particle_weights = dict(sorted_particles[:num_particles])

    weights = utils.apply_temp(
      np.array(list(new_particle_weights.values()), dtype=np.float32),
      temperature=temperature,
    )

    return dict(zip(new_particle_weights.keys(), weights))

  def _rebuild_particles(self) -> None:
    """Rebuild particles using the stored observation history."""
    if not self._history:
      self._particle_weights = {}
      return

    logger.debug(
      f"Rebuilding particles with history of {len(self._history)} steps and {len(self._particle_weights)} existing particles."
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
        particle_weights = self._history[start_idx].particle_weights
      else:
        particle_weights = {INITIAL_STATE_SERIALIZED: 1.0}
        start_idx = 0

      logger.debug(
        f"Rebuild attempt {attempt + 1}/{self.rebuild_tries} starting from index {start_idx}."
      )
      history_scale = (
        1.0 + (len(self._history) - start_idx) / self.game.max_game_length()
      )
      attempt_scale = 1.0 + attempt / self.rebuild_tries
      num_particles = int(
        self.max_num_particles * history_scale * attempt_scale
      )
      matches_per_particle = min(
        int(self.max_matches_per_particle * history_scale * attempt_scale),
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
        if not particle_weights:
          ok = False
          break

        if rec.actor_is_ai:
          particle_weights = self._resample_particles_for_ai(
            rec, particle_weights, num_particles=num_particles
          )
        else:
          particle_weights = self._resample_particles_for_opponent(
            rec.ai_obs_after,
            particle_weights,
            matches_per_particle,
            num_particles=num_particles,
            temperature=temperature,
          )

        if idx in self._checkpoint_indices and (
          not rec.particle_weights
          or len(rec.particle_weights) < len(particle_weights)
        ):
          logger.debug(
            f"Updating checkpoint at index {idx} with {len(particle_weights)} particles."
          )
          rec.particle_weights = particle_weights.copy()

      if ok and particle_weights:
        if attempt > 0:
          checkpoint_idx = len(self._history) - 1
          self._checkpoint_indices.add(checkpoint_idx)
          rec = self._history[checkpoint_idx]
          rec.particle_weights = particle_weights.copy()
          logger.debug(
            f"Final checkpoint updated with {len(particle_weights)} particles."
          )

        self._particle_weights = particle_weights
        return
