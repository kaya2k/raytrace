import numpy as np
import numba
from utils import EPSILON
from .shape import Shape


class Sphere(Shape):
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = 0.0

        self.DIFF_C = 0.8
        self.SPEC_C = 0.2
        self.SPEC_K = 32

    def intersect(self, origin, direction):
        return sphere_intersect(origin, direction, self.center, self.radius)

    def get_normal(self, point):
        return sphere_get_normal(point, self.center)


@numba.njit(fastmath=True)
def sphere_intersect(origin, direction, center, radius):
    oc = origin - center
    b = direction[0] * oc[0] + direction[1] * oc[1] + direction[2] * oc[2]
    c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius

    disc = b * b - c
    if disc < 0.0:
        return np.inf

    sqrt_disc = np.sqrt(disc)
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc

    if t0 > EPSILON and t1 > EPSILON:
        return t0 if t0 < t1 else t1
    if t0 > EPSILON:
        return t0
    if t1 > EPSILON:
        return t1
    return np.inf


@numba.njit(fastmath=True)
def sphere_get_normal(point, center):
    nx = point[0] - center[0]
    ny = point[1] - center[1]
    nz = point[2] - center[2]

    length_sq = nx * nx + ny * ny + nz * nz
    if length_sq <= 0.0:
        return np.array((0.0, 0.0, 0.0))

    inv_len = 1.0 / np.sqrt(length_sq)
    return np.array((nx * inv_len, ny * inv_len, nz * inv_len))
