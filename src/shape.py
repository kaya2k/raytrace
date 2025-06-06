import numpy as np
from abc import ABC, abstractmethod
from utils import EPSILON


class Shape(ABC):
    @abstractmethod
    def get_normal(self, point):
        raise NotImplementedError("Subclasses must implement get_normal method")

    @abstractmethod
    def intersect(self, origin, direction):
        raise NotImplementedError("Subclasses must implement intersect method")


class Cube(Shape):
    def __init__(self, center, feature, color):
        self.center = np.array(center)
        self.feature = np.array(feature)
        self.color = np.array(color)

        self.diffuse_c = 0.8
        self.specular_c = 0.2
        self.specular_k = 32

        self.min_bound = self.center - self.feature / 2
        self.max_bound = self.center + self.feature / 2

    def intersect(self, origin, direction):
        inv_dir = 1.0 / np.where(direction != 0, direction, EPSILON)
        t_min = (self.min_bound - origin) * inv_dir
        t_max = (self.max_bound - origin) * inv_dir

        t_near = np.max(np.minimum(t_min, t_max))
        t_far = np.min(np.maximum(t_min, t_max))

        if t_near > t_far or t_far < 0:
            return np.inf

        return t_near if t_near >= 0 else t_far

    def get_normal(self, point):
        diff = point - self.center
        half = self.feature / 2

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
