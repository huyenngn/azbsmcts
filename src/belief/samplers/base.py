from __future__ import annotations

import typing as t


class DeterminizationSampler(t.Protocol):
  """Protocol for samplers that generate determinizations from belief states."""

  def sample(self) -> str:
    """Return a serialized determinized state."""
    ...

  def sample_with_prior(self) -> tuple[str, float]:
    """Return (serialized determinized state, prior probability) tuple."""
    ...
