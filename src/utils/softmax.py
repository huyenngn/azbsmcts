import numpy as np


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Compute softmax over a 1D array with numerical stability."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    finite_mask = np.isfinite(x)
    if not np.any(finite_mask):
        return np.ones_like(x) / x.size

    safe_x = np.where(finite_mask, x, -np.inf)
    x_max = np.max(safe_x)
    exp_x = np.exp(safe_x - x_max)
    exp_x[~finite_mask] = 0.0

    total = np.sum(exp_x)
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(x) / x.size
    return exp_x / total
