"""
Phantom Go API backend.

Single-client server for playing against azbsmcts, bsmcts, or random agents.
"""

import enum
import logging
import os
import pathlib
import random
import re
import urllib.request

import fastapi
import pydantic
import pyspiel
import uvicorn

from agents import AZBSMCTSAgent, BSMCTSAgent
from belief.samplers.particle import (
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)
from utils import utils

DEMO_MODEL_URL = "https://github.com/huyenngn/alphaghost/releases/download/demo-model/demo_model.pt"
DEFAULT_DEMO_MODEL_PATH = pathlib.Path("models/demo_model.pt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phantom_go_api")

app = fastapi.FastAPI()

app.state.game_state = None
app.state.human_id = 0
app.state.ai_id = 1
app.state.agent = None
app.state.particle = None


class PlayerColor(enum.Enum):
    Black = 0
    White = 1


class StartGameRequest(pydantic.BaseModel):
    player_id: int
    policy: str


class MakeMoveRequest(pydantic.BaseModel):
    action: int


class PreviousMoveInfo(pydantic.BaseModel):
    player: PlayerColor = PlayerColor.Black
    was_observational: bool = False
    was_pass: bool = False
    captured_stones: int = 0


class GameStateResponse(pydantic.BaseModel):
    observation: str = ""
    previous_move_infos: list[PreviousMoveInfo] = []
    is_terminal: bool = False
    returns: list[float] = []


def _ensure_demo_model(path: pathlib.Path) -> str:
    """Download demo model if not present."""
    utils.ensure_dir(path.parent)
    if path.exists():
        return str(path)

    logger.info("Downloading demo model to %s", path)
    try:
        urllib.request.urlretrieve(DEMO_MODEL_URL, str(path))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Failed to download demo model: {e}"
        )
    return str(path)


def _build_agent(game: pyspiel.Game, policy: str, ai_id: int):
    """Create agent and particle sampler for the given policy."""
    if policy == "random":
        logger.info("Using random policy")
        return None, None

    num_actions = game.num_distinct_actions()
    particle = ParticleBeliefSampler(
        game=game,
        ai_id=ai_id,
        num_particles=32,
        opp_tries_per_particle=8,
        rebuild_max_tries=200,
        seed=1234 + ai_id,
    )
    sampler = ParticleDeterminizationSampler(particle)

    if policy == "bsmcts":
        logger.info("Using BS-MCTS policy")
        agent = BSMCTSAgent(
            player_id=ai_id,
            num_actions=num_actions,
            sampler=sampler,
            T=4,
            S=2,
            seed=1 + ai_id,
        )
        return agent, particle

    if policy == "azbsmcts":
        logger.info("Using AZ-BS-MCTS policy")
        model_path = _ensure_demo_model(
            pathlib.Path(
                os.environ.get("AZ_MODEL_PATH", str(DEFAULT_DEMO_MODEL_PATH))
            )
        )
        agent = AZBSMCTSAgent(
            player_id=ai_id,
            num_actions=num_actions,
            obs_size=game.observation_tensor_size(),
            sampler=sampler,
            T=4,
            S=2,
            seed=3 + ai_id,
            device=os.environ.get("AZ_DEVICE", "cpu"),
            model_path=model_path,
        )
        return agent, particle

    return None, None


def _parse_move_info() -> PreviousMoveInfo | None:
    """Parse move info from observation string."""
    if app.state.game_state is None or app.state.game_state.is_terminal():
        return None

    obs = app.state.game_state.observation_string(app.state.human_id)[-90:]
    match = re.search(
        r"Previous move was (valid|observational)"
        r"(?:\s+and was a (pass)|\s+In previous move (\d+) stones were captured)?",
        obs,
    )
    if not match:
        return None

    return PreviousMoveInfo(
        was_observational=match.group(1) == "observational",
        was_pass=match.group(2) is not None,
        captured_stones=int(match.group(3)) if match.group(3) else 0,
    )


def _apply_action(player: int, action: int) -> None:
    """Apply action and update belief sampler."""
    app.state.game_state.apply_action(action)
    if app.state.particle is not None:
        app.state.particle.step(
            actor=player, action=action, real_state_after=app.state.game_state
        )


def _play_ai_turns() -> list[PreviousMoveInfo]:
    """Execute all consecutive AI turns."""
    move_infos = []
    while (
        app.state.game_state is not None
        and not app.state.game_state.is_terminal()
        and app.state.game_state.current_player() == app.state.ai_id
    ):
        if app.state.agent is None:
            action = random.choice(app.state.game_state.legal_actions())
        else:
            action = app.state.agent.select_action(app.state.game_state)

        logger.info("AI plays action %d", action)
        _apply_action(app.state.ai_id, action)

        info = _parse_move_info()
        if info:
            info.player = PlayerColor(app.state.ai_id)
            move_infos.append(info)

    return move_infos


def _make_response(move_infos: list[PreviousMoveInfo]) -> GameStateResponse:
    """Build response from current state."""
    return GameStateResponse(
        observation=app.state.game_state.observation_string(app.state.human_id)
        if not app.state.game_state.is_terminal()
        else "",
        previous_move_infos=move_infos,
        is_terminal=app.state.game_state.is_terminal(),
        returns=list(app.state.game_state.returns()),
    )


@app.get("/")
def root():
    """API info endpoint."""
    return {"name": "ALPHAGhOst", "description": "Phantom Go AI backend"}


@app.post("/start")
def start_game(request: StartGameRequest) -> GameStateResponse:
    """Start a new game, replacing any existing game."""

    if request.player_id not in (0, 1):
        raise fastapi.HTTPException(
            status_code=400, detail="player_id must be 0 or 1"
        )

    logger.info(
        "Starting new game: human=%d, policy=%s",
        request.player_id,
        request.policy,
    )

    game = pyspiel.load_game("phantom_go", {})
    app.state.game_state = game.new_initial_state()
    app.state.human_id = request.player_id
    app.state.ai_id = 1 - app.state.human_id
    app.state.agent, app.state.particle = _build_agent(
        game, request.policy, app.state.ai_id
    )

    move_infos = _play_ai_turns()
    return _make_response(move_infos)


@app.post("/step")
def make_move(request: MakeMoveRequest) -> GameStateResponse:
    """Apply human move and execute AI response."""
    if app.state.game_state is None:
        raise fastapi.HTTPException(status_code=400, detail="No active game")
    if app.state.game_state.is_terminal():
        raise fastapi.HTTPException(status_code=400, detail="Game is over")
    if app.state.game_state.current_player() != app.state.human_id:
        raise fastapi.HTTPException(status_code=400, detail="Not your turn")
    if request.action not in app.state.game_state.legal_actions():
        raise fastapi.HTTPException(status_code=400, detail="Illegal action")

    logger.info("Human plays action %d", request.action)
    _apply_action(app.state.human_id, request.action)

    move_infos = []
    info = _parse_move_info()
    if info:
        info.player = PlayerColor(app.state.human_id)
        move_infos.append(info)

    move_infos.extend(_play_ai_turns())

    if app.state.game_state.is_terminal():
        logger.info("Game ended: returns=%s", app.state.game_state.returns())

    return _make_response(move_infos)


def main():
    """Run the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
