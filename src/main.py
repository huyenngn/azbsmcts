import enum
import logging
import os
import random
import re
import urllib.request
import uuid
from pathlib import Path

import pydantic
import pyspiel
import uvicorn
from fastapi import FastAPI, HTTPException

from agents import AZBSMCTSAgent, BSMCTSAgent
from belief.samplers.particle import (
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)

DEMO_MODEL_URL = "https://github.com/huyenngn/alphaghost/releases/download/demo-model/demo_model.pt"
DEFAULT_DEMO_MODEL_PATH = Path("models/demo_model.pt")


logger = logging.getLogger("phantom_go_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.state.games = {}


class StartGameRequest(pydantic.BaseModel):
    player_id: int
    policy: str


class MakeMoveRequest(pydantic.BaseModel):
    game_id: str
    action: int


class PlayerColor(enum.Enum):
    Black = 0
    White = 1


class PreviousMoveInfo(pydantic.BaseModel):
    player: PlayerColor = PlayerColor.Black
    was_observational: bool = False
    was_pass: bool = False
    captured_stones: int = 0


class GameStateResponse(pydantic.BaseModel):
    game_id: str = ""
    observation: str = ""
    previous_move_infos: list[PreviousMoveInfo] = []
    is_terminal: bool = False
    returns: list[float] = []


def ensure_demo_model(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return str(path)

    logger.info("AZ model not found at %s. Downloading demo model...", path)
    try:
        urllib.request.urlretrieve(DEMO_MODEL_URL, str(path))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download demo model from release 'demo-model': {e}",
        )
    return str(path)


def _build_agent(game: pyspiel.Game, policy: str, ai_id: int):
    if policy == "random":
        return None, None

    num_actions = game.num_distinct_actions()
    obs_size = len(game.new_initial_state().observation_tensor())

    # Non-cheating belief sampler for THIS AI
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
        requested_path = Path(
            os.environ.get("AZ_MODEL_PATH", str(DEFAULT_DEMO_MODEL_PATH))
        )
        model_path = ensure_demo_model(requested_path)

        agent = AZBSMCTSAgent(
            player_id=ai_id,
            num_actions=num_actions,
            obs_size=obs_size,
            sampler=sampler,
            T=4,
            S=2,
            seed=3 + ai_id,
            device=os.environ.get("AZ_DEVICE", "cpu"),
            model_path=model_path,
        )
        return agent, particle

    return None, None


def _get_ai_action(session: dict) -> int:
    state = session["state"]
    agent = session["agent"]

    if agent is None:
        return random.choice(state.legal_actions())
    return agent.select_action(state)


def _get_move_info(session: dict) -> PreviousMoveInfo | None:
    state = session["state"]
    human_id = session["human_id"]

    if state.is_terminal():
        return None

    observation_str = state.observation_string(human_id)[-90:]
    match = re.search(
        r"Previous move was (valid|observational)(?:\s+and was a (pass)|\s+In previous move (\d+) stones were captured)?",
        observation_str,
    )
    if not match:
        return None

    was_observational = match.group(1) == "observational"
    was_pass = match.group(2) is not None
    captured_stones = int(match.group(3)) if match.group(3) else 0

    return PreviousMoveInfo(
        was_observational=was_observational,
        was_pass=was_pass,
        captured_stones=captured_stones,
    )


def _sync_beliefs_after_real_move(session: dict, actor: int, action: int):
    """
    Update the particle belief sampler using ONLY observation strings.
    Opponent actions are passed but IGNORED by the sampler (non-cheating).
    """
    particle = session.get("particle")
    if particle is None:
        return
    particle.step(
        actor=actor, action=action, real_state_after=session["state"]
    )


def _play_ai_turns(session: dict) -> list[PreviousMoveInfo]:
    state = session["state"]
    ai_id = session["ai_id"]
    move_infos = []

    while not state.is_terminal() and state.current_player() == ai_id:
        actor = state.current_player()
        action = _get_ai_action(session)
        state.apply_action(action)
        _sync_beliefs_after_real_move(session, actor=actor, action=action)

        info = _get_move_info(session)
        if info:
            info.player = PlayerColor(ai_id)
            move_infos.append(info)

    return move_infos


@app.get("/")
def root():
    return {"name": "ALPHAGhOst", "description": "A Phantom Go game AI"}


@app.post("/start")
def start_game(request: StartGameRequest) -> GameStateResponse:
    if request.player_id not in (0, 1):
        raise HTTPException(status_code=400, detail="player_id must be 0 or 1")

    game = pyspiel.load_game("phantom_go", {})
    state = game.new_initial_state()

    human_id = request.player_id
    ai_id = 1 - human_id

    agent, particle = _build_agent(game, request.policy, ai_id)

    game_id = str(uuid.uuid4())
    session = {
        "state": state,
        "human_id": human_id,
        "ai_id": ai_id,
        "agent": agent,
        "particle": particle,
    }
    app.state.games[game_id] = session

    # AI plays first if human is white
    move_infos = _play_ai_turns(session)

    return GameStateResponse(
        game_id=game_id,
        observation=state.observation_string(human_id)
        if not state.is_terminal()
        else "",
        previous_move_infos=move_infos,
        is_terminal=state.is_terminal(),
        returns=state.returns(),
    )


@app.post("/step")
def make_move(request: MakeMoveRequest) -> GameStateResponse:
    session = app.state.games.get(request.game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown game_id")

    state = session["state"]
    human_id = session["human_id"]

    if state.is_terminal():
        raise HTTPException(status_code=400, detail="Game is over")

    if state.current_player() != human_id:
        raise HTTPException(status_code=400, detail="Not your turn")

    if request.action not in state.legal_actions():
        raise HTTPException(status_code=400, detail="Illegal action")

    actor = state.current_player()
    state.apply_action(request.action)
    _sync_beliefs_after_real_move(session, actor=actor, action=request.action)

    move_infos = []
    info = _get_move_info(session)
    if info:
        info.player = PlayerColor(human_id)
        move_infos.append(info)

    ai_move_infos = _play_ai_turns(session)
    move_infos.extend(ai_move_infos)

    if state.is_terminal():
        app.state.games.pop(request.game_id, None)

    return GameStateResponse(
        game_id=request.game_id,
        observation=state.observation_string(human_id)
        if not state.is_terminal()
        else "",
        previous_move_infos=move_infos,
        is_terminal=state.is_terminal(),
        returns=state.returns(),
    )


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
