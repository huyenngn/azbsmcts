import numpy as np


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Compute softmax over a 1D array with numerical stability."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
