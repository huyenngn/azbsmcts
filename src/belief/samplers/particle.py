from __future__ import annotations

import dataclasses
import math
import random
from collections import abc as cabc

import numpy as np

import openspiel

# Type alias for opponent policy function: state -> action probabilities over full action space.
OpponentPolicy = cabc.Callable[[openspiel.State], np.ndarray]


@dataclasses.dataclass(frozen=True)
class _StepRecord:
  """Record of one real move from the AI player's perspective."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: str


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance.

  Key properties:
  - Maintains a set of *unique* particles (states) consistent with the AI player's
    observation history (as captured by observation_string(ai_id)).
  - On opponent turns, it can branch into up to K matching successor actions per particle.
  - When branching creates too many candidates, it resamples back to num_particles
    (weighted by opponent policy within the matching set when available).
  - Never hard-crashes training: sample() will attempt rebuild and then fall back.
  """

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    num_particles: int = 32,
    # Max number of matching opponent actions to keep per particle per opponent step.
    max_matching_opp_actions: int = 8,
    # How many rebuild attempts to try before falling back.
    rebuild_max_tries: int = 50,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
  ):
    if num_particles <= 0:
      raise ValueError("num_particles must be > 0")
    if max_matching_opp_actions <= 0:
      raise ValueError("max_matching_opp_actions must be > 0")
    if rebuild_max_tries <= 0:
      raise ValueError("rebuild_max_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.num_particles = num_particles
    self.max_matching_opp_actions = max_matching_opp_actions
    self.rebuild_max_tries = rebuild_max_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy

    self._history: list[_StepRecord] = []
    # Unique particles by serialize() key.
    self._particles: dict[str, openspiel.State] = {}

  # -------------------------
  # Public API
  # -------------------------

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particles.clear()

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
    )
    self._history.append(rec)

    # If we have no particles yet, build from scratch.
    if not self._particles:
      self._rebuild_particles()
      return

    if rec.actor_is_ai:
      self._update_for_ai_move(rec)
    else:
      self._update_for_opponent_move(rec)

    # If we collapsed, try rebuild.
    if not self._particles:
      self._rebuild_particles()

  def sample(self) -> openspiel.State:
    """Sample a particle consistent with current observation history.

    This function MUST NOT crash training. If belief collapses:
    - attempt rebuild
    - if still empty, fall back to a fresh initial state (uninformed determinization)
    """
    if not self._history:
      return self.game.new_initial_state()

    if not self._particles:
      self._rebuild_particles()

    if not self._particles:
      raise RuntimeError(
        "ParticleDeterminizationSampler: belief collapsed completely; "
        "unable to rebuild particles consistent with observation history."
      )

    return self.rng.choice(list(self._particles.values())).clone()

  # -------------------------
  # Internals: observation + ordering
  # -------------------------

  def _ai_obs(self, state: openspiel.State) -> str:
    return state.observation_string(self.ai_id)

  def _get_opponent_actions(self, state: openspiel.State) -> list[int]:
    """Get legal opponent actions in a deterministic order (policy) or random permutation (no policy)."""
    legal = state.legal_actions()
    if not legal:
      return []

    # No policy: uniform random permutation (no duplicates).
    if self.opponent_policy is None:
      return self.rng.sample(legal, len(legal))

    probs = self.opponent_policy(state)
    # Extract and normalize only legal probs.
    legal_probs = np.array([probs[a] for a in legal], dtype=np.float64)
    legal_probs = np.maximum(legal_probs, 0.0)
    total = float(legal_probs.sum())
    if total <= 0.0:
      return self.rng.sample(legal, len(legal))

    legal_probs /= total
    # Sort legal actions by descending probability (no duplicates).
    return [
      a
      for a, _ in sorted(
        zip(legal, legal_probs), key=lambda x: x[1], reverse=True
      )
    ]

  def _get_action_weight(self, state: openspiel.State, action: int) -> float:
    """Policy weight for an action in a given state (clipped to non-negative)."""
    if self.opponent_policy is None:
      return 1.0
    probs = self.opponent_policy(state)
    if action < 0 or action >= len(probs):
      return 0.0
    w = float(probs[action])
    return w if w > 0.0 else 0.0

  # -------------------------
  # Internals: particle updates
  # -------------------------

  def _update_for_ai_move(self, rec: _StepRecord) -> None:
    assert rec.actor_is_ai and rec.ai_action is not None
    updated: dict[str, openspiel.State] = {}
    for p in self._particles.values():
      p2 = p.clone()
      if rec.ai_action not in p2.legal_actions():
        continue
      p2.apply_action(rec.ai_action)
      if self._ai_obs(p2) == rec.ai_obs_after:
        updated[p2.serialize()] = p2
    self._particles = updated
    self._downsample_particles_uniform()

  def _update_for_opponent_move(self, rec: _StepRecord) -> None:
    assert not rec.actor_is_ai
    candidates: list[tuple[openspiel.State, float]] = []
    for p in self._particles.values():
      # For each particle, generate up to K matching successor states.
      candidates.extend(self._matching_opponent_children(p, rec.ai_obs_after))

    self._particles = self._resample_unique_candidates(candidates)

  def _matching_opponent_children(
    self, particle: openspiel.State, target_obs: str
  ) -> list[tuple[openspiel.State, float]]:
    """Generate up to K matching opponent-successor states for a single particle.

    Returns list of (child_state, weight).
    Weight is proportional to opponent policy prob for the chosen action within legal actions.
    """
    p2 = particle.clone()
    if not p2.legal_actions():
      return []

    opp_actions = self._get_opponent_actions(p2)
    if not opp_actions:
      return []

    matches: list[tuple[openspiel.State, float]] = []
    seen: set[int] = set()

    # We scan actions in the provided order, collecting up to K matches.
    for a in opp_actions:
      if a in seen:
        continue
      seen.add(a)

      p3 = p2.clone()
      p3.apply_action(a)
      if self._ai_obs(p3) != target_obs:
        continue

      w = self._get_action_weight(p2, a)
      # If policy is present but gives 0 everywhere, we still want diversity.
      if w <= 0.0:
        w = 1e-12

      matches.append((p3, w))
      if len(matches) >= self.max_matching_opp_actions:
        break

    return matches

  # -------------------------
  # Internals: resampling / dedupe
  # -------------------------

  def _downsample_particles_uniform(self) -> None:
    """Uniformly downsample current unique particles to num_particles (if needed)."""
    if len(self._particles) <= self.num_particles:
      return
    keys = list(self._particles.keys())
    keep = set(self.rng.sample(keys, self.num_particles))
    self._particles = {k: self._particles[k] for k in keep}

  def _resample_unique_candidates(
    self, candidates: list[tuple[openspiel.State, float]]
  ) -> dict[str, openspiel.State]:
    """Deduplicate candidates by serialize key, aggregate weights, then resample to num_particles."""
    if not candidates:
      return {}

    # Deduplicate and aggregate weights per state key.
    by_key: dict[str, tuple[openspiel.State, float]] = {}
    for s, w in candidates:
      k = s.serialize()
      if k in by_key:
        # Keep one representative state object; add weight.
        rep, w0 = by_key[k]
        by_key[k] = (rep, w0 + float(w))
      else:
        by_key[k] = (s, float(w))

    items = list(by_key.items())  # (key, (state, weight))
    if len(items) <= self.num_particles:
      return {k: st for k, (st, _) in items}

    # Weighted sampling without replacement using Efraimidis–Spirakis keys:
    # For weight w, sample key u^(1/w). Larger key => higher chance.
    scored: list[tuple[float, str, openspiel.State]] = []
    for k, (st, w) in items:
      w = float(w)
      if not math.isfinite(w) or w <= 0.0:
        w = 1e-12
      u = self.rng.random()
      # Avoid log(0) / 0^inf weirdness.
      u = max(u, 1e-12)
      score = u ** (1.0 / w)
      scored.append((score, k, st))

    scored.sort(reverse=True, key=lambda x: x[0])
    chosen = scored[: self.num_particles]
    return {k: st for _, k, st in chosen}

  # -------------------------
  # Rebuild
  # -------------------------

  def _rebuild_particles(self) -> None:
    """Rebuild particles from scratch using the stored observation history.

    This is a bounded-attempt particle-filter-like forward pass:
    - Start from initial state particle(s)
    - For each record in history:
        - AI move: deterministic filter
        - Opp move: branch into up to K matching actions per particle, then resample
    If rebuild fails (particles become empty), we retry from scratch a few times
    (retries mainly matter because resampling is stochastic).
    """
    if not self._history:
      self._particles = {}
      return

    # Preserve current K and optionally escalate on later attempts.
    base_k = self.max_matching_opp_actions

    for attempt in range(self.rebuild_max_tries):
      # Mild escalation every few attempts if we're repeatedly failing.
      if attempt > 0 and attempt % 10 == 0:
        self.max_matching_opp_actions = min(base_k * 2, 64)

      particles: dict[str, openspiel.State] = {}
      s0 = self.game.new_initial_state()
      particles[s0.serialize()] = s0

      ok = True
      for rec in self._history:
        if not particles:
          ok = False
          break

        if rec.actor_is_ai:
          updated: dict[str, openspiel.State] = {}
          assert rec.ai_action is not None
          for p in particles.values():
            p2 = p.clone()
            if rec.ai_action not in p2.legal_actions():
              continue
            p2.apply_action(rec.ai_action)
            if self._ai_obs(p2) == rec.ai_obs_after:
              updated[p2.serialize()] = p2
          particles = updated
          # Keep size under control even on AI turns.
          if len(particles) > self.num_particles:
            # Uniform downsample is fine here; no policy signal on AI moves.
            keys = list(particles.keys())
            keep = set(self.rng.sample(keys, self.num_particles))
            particles = {k: particles[k] for k in keep}
          continue

        # Opponent move: branch + resample.
        candidates: list[tuple[openspiel.State, float]] = []
        for p in particles.values():
          candidates.extend(
            self._matching_opponent_children(p, rec.ai_obs_after)
          )
        particles = self._resample_unique_candidates(candidates)

      if ok and particles:
        self._particles = particles
        # Restore K to base for normal operation.
        self.max_matching_opp_actions = base_k
        return

    # If we get here, rebuild failed. Keep empty and let sample() fall back safely.
    self._particles = {}
    self.max_matching_opp_actions = base_k
