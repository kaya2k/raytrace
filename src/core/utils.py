import math
from numba import cuda


@cuda.jit(device=True)
def dot3(ax, ay, az, bx, by, bz) -> float:
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True)
def normalize3(x, y, z):
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return 0.0, 0.0, 0.0
    inv = 1.0 / length
    return x * inv, y * inv, z * inv
