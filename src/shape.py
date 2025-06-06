import numpy as np
import numba
from abc import ABC, abstractmethod
from utils import EPSILON


class Shape(ABC):
    def __init__(self):
        self.center = np.zeros(3)
        self.feature = np.zeros(3)
        self.color = np.zeros(3)

        self.DIFF_C = 0.0
        self.SPEC_C = 0.0
        self.SPEC_K = 0.0

    @abstractmethod
    def intersect(self, origin, direction):
        raise NotImplementedError("Subclasses must implement intersect method")

    @abstractmethod
    def get_normal(self, point):
        raise NotImplementedError("Subclasses must implement get_normal method")


class Cube(Shape):
    def __init__(self, center, feature, color):
        self.center = np.array(center)
        self.feature = np.array(feature)
        self.color = np.array(color)

        self.DIFF_C = 0.8
        self.SPEC_C = 0.2
        self.SPEC_K = 32

        self.min_bound = self.center - self.feature / 2
        self.max_bound = self.center + self.feature / 2

    def intersect(self, origin, direction):
        return cube_intersect(origin, direction, self.min_bound, self.max_bound)

    def get_normal(self, point):
        return cube_get_normal(point, self.center, self.feature)


@numba.njit
def cube_intersect(origin, direction, min_bound, max_bound):
    inv_dir = 1.0 / np.where(direction != 0, direction, EPSILON)
    t_min = (min_bound - origin) * inv_dir
    t_max = (max_bound - origin) * inv_dir
    t_near = np.max(np.minimum(t_min, t_max))
    t_far = np.min(np.maximum(t_min, t_max))
    if t_near > t_far or t_far < 0:
        return np.inf
    return t_near if t_near >= 0 else t_far


@numba.njit
def cube_get_normal(point, center, feature):
    diff = point - center
    half = feature / 2
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
