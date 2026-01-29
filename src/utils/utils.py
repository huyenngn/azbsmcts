import pathlib

import pyspiel


def ensure_dir(p: pathlib.Path) -> None:
    """Create directory and parents if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def clone_state(state: pyspiel.State) -> pyspiel.State:
    """Create a deep copy of a pyspiel State via serialization."""
    return state.get_game().deserialize_state(state.serialize())
