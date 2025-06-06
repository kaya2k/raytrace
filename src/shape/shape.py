import numpy as np
from abc import ABC, abstractmethod


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
