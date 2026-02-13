from __future__ import annotations

import typing as t


class DeterminizationSampler(t.Protocol):
  """Protocol for samplers that generate determinizations from belief states."""

  def sample(self) -> str:
    """Return a serialized determinized state."""
    ...

  def sample_with_prior(self) -> tuple[str, float]:
    """Return (serialized state, prior probability P(γ)).

    The prior reflects how likely this determinization is given the
    opponent model.  Samplers without an opponent model should return
    ``P(γ) = 1.0``.
    """
    ...
