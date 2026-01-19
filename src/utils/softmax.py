from __future__ import annotations

import numpy as np


def softmax_np(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    t = max(1e-8, float(temp))
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        # should not happen, but avoid NaNs
        out = np.zeros_like(e)
        out[np.argmax(z)] = 1.0
        return out
    return e / s
