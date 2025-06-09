import math
from numba import cuda
from . import EPSILON


@cuda.jit(device=True)
def intersect_sphere_device(
    origin_x,
    origin_y,
    origin_z,
    dir_x,
    dir_y,
    dir_z,
    center_x,
    center_y,
    center_z,
    radius,
) -> float:
    # Vector from sphere center to ray origin
    ox = origin_x - center_x
    oy = origin_y - center_y
    oz = origin_z - center_z
    # Coefficients for quadratic equation: t^2 + 2*b*t + c = 0
    # Here b = dot(direction, (origin-center)), c = ||origin-center||^2 - r^2
    b = ox * dir_x + oy * dir_y + oz * dir_z
    c = ox * ox + oy * oy + oz * oz - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return math.inf  # no real roots, ray misses the sphere
    sqrt_disc = math.sqrt(disc)
    # Compute the two intersection distances
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc
    # We want the smallest positive t (in front of the origin)
    valid_t0 = t0 > EPSILON
    valid_t1 = t1 > EPSILON
    if valid_t0 and valid_t1:
        return t0 if t0 < t1 else t1
    if valid_t0:
        return t0
    if valid_t1:
        return t1
    return math.inf  # both intersections are behind the origin or too close


@cuda.jit(device=True)
def normal_sphere_device(point_x, point_y, point_z, center_x, center_y, center_z):
    # Compute the normal as the normalized vector from sphere center to hit point
    nx = point_x - center_x
    ny = point_y - center_y
    nz = point_z - center_z
    length_sq = nx * nx + ny * ny + nz * nz
    if length_sq <= 0.0:
        return 0.0, 0.0, 0.0
    inv_len = 1.0 / math.sqrt(length_sq)
    return nx * inv_len, ny * inv_len, nz * inv_len
