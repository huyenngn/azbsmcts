import pathlib

import numpy as np


def ensure_dir(p: pathlib.Path) -> None:
  """Create directory and parents if they don't exist."""
  p.mkdir(parents=True, exist_ok=True)


def apply_temp(probs: np.ndarray, temperature: float) -> np.ndarray:
  """Apply temperature to a probability distribution; values > 1 flatten it to
  explore rare actions more; values < 1 sharpen it.
  """
  if temperature != 1.0:
    probs = np.power(probs, 1.0 / temperature)
    probs /= probs.sum()
  return probs
