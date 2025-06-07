import numpy as np
from numba import njit


@njit(fastmath=True, inline="always")
def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath=True, inline="always")
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.sqrt(dot(v, v))
    return np.array((v[0] / n, v[1] / n, v[2] / n), dtype=np.float64)
