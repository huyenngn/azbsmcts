import pathlib


def ensure_dir(p: pathlib.Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
