from __future__ import annotations

import random

import pyspiel

from agents import AZBSMCTSAgent, BSMCTSAgent
from belief.samplers.particle import (
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)
from scripts.common.config import SamplerConfig, SearchConfig
from scripts.common.seeding import derive_seed


def make_belief_sampler(
    *,
    game: pyspiel.Game,
    player_id: int,
    base_seed: int,
    sampler_cfg: SamplerConfig,
    run_id: str,
    purpose: str,
    game_idx: int | None,
) -> ParticleBeliefSampler:
    seed = derive_seed(
        base_seed,
        purpose=f"{purpose}/belief",
        run_id=run_id,
        game_idx=game_idx,
        player_id=player_id,
    )
    return ParticleBeliefSampler(
        game=game,
        ai_id=player_id,
        num_particles=sampler_cfg.num_particles,
        opp_tries_per_particle=sampler_cfg.opp_tries_per_particle,
        rebuild_max_tries=sampler_cfg.rebuild_max_tries,
        seed=seed,
    )


def make_agent(
    *,
    kind: str,
    player_id: int,
    game: pyspiel.Game,
    search_cfg: SearchConfig,
    sampler_cfg: SamplerConfig,
    base_seed: int,
    run_id: str,
    purpose: str,
    device: str,
    model_path: str | None,
    net=None,  # optional for training self-play
    game_idx: int | None,
):
    if kind == "random":
        return None, None

    particle = make_belief_sampler(
        game=game,
        player_id=player_id,
        base_seed=base_seed,
        sampler_cfg=sampler_cfg,
        run_id=run_id,
        purpose=purpose,
        game_idx=game_idx,
    )
    sampler = ParticleDeterminizationSampler(particle)

    num_actions = game.num_distinct_actions()
    agent_seed = derive_seed(
        base_seed,
        purpose=f"{purpose}/agent",
        run_id=run_id,
        game_idx=game_idx,
        player_id=player_id,
    )

    if kind == "bsmcts":
        return (
            BSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                sampler=sampler,
                T=search_cfg.T,
                S=search_cfg.S,
                seed=agent_seed,
            ),
            particle,
        )

    if kind == "azbsmcts":
        return (
            AZBSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                obs_size=game.observation_tensor_size(),
                sampler=sampler,
                T=search_cfg.T,
                S=search_cfg.S,
                seed=agent_seed,
                device=device,
                model_path=model_path,
                net=net,
            ),
            particle,
        )

    raise ValueError(f"Unknown agent kind: {kind}")


def select_action(
    kind: str, agent, state: pyspiel.State, rng: random.Random
) -> int:
    if kind == "random" or agent is None:
        return rng.choice(state.legal_actions())
    return agent.select_action(state)
