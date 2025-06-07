import numpy as np
from numba import njit
from . import EPSILON


@njit(fastmath=True, inline="always")
def intersect_cube(
    origin: np.ndarray,
    direction: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> float:
    inv_dir = 1.0 / np.where(direction != 0, direction, EPSILON)
    t_min = (min_bound - origin) * inv_dir
    t_max = (max_bound - origin) * inv_dir
    t_near = np.max(np.minimum(t_min, t_max))
    t_far = np.min(np.maximum(t_min, t_max))
    if t_near > t_far or t_far < 0:
        return np.inf
    return t_near if t_near >= 0 else t_far


@njit(fastmath=True, inline="always")
def normal_cube(
    point: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
) -> np.ndarray:
    diff = point - center
    half = size / 2
    for i in range(3):
        if abs(diff[i] - half[i]) < EPSILON:
            normal = np.zeros(3)
            normal[i] = 1
            return normal
        elif abs(diff[i] + half[i]) < EPSILON:
            normal = np.zeros(3)
            normal[i] = -1
            return normal
    raise ValueError("Point is not on the surface of the cube")
