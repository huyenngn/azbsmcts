from __future__ import annotations

import typing as t


class DeterminizationSampler(t.Protocol):
  """Protocol for samplers that generate determinizations from belief states."""

  def sample(self) -> str: ...
