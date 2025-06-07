import math
from numba import cuda

EPSILON = 1e-6  # small constant to avoid division by zero and self-intersections


@cuda.jit(device=True)
def intersect_cube_device(
    origin_x,
    origin_y,
    origin_z,
    dir_x,
    dir_y,
    dir_z,
    min_x,
    min_y,
    min_z,
    max_x,
    max_y,
    max_z,
) -> float:
    # Compute inverse direction, avoiding division by zero
    inv_dir_x = 1.0 / (dir_x if abs(dir_x) > EPSILON else EPSILON)
    inv_dir_y = 1.0 / (dir_y if abs(dir_y) > EPSILON else EPSILON)
    inv_dir_z = 1.0 / (dir_z if abs(dir_z) > EPSILON else EPSILON)
    # Compute intersection distances for each axis slab
    t1_x = (min_x - origin_x) * inv_dir_x
    t2_x = (max_x - origin_x) * inv_dir_x
    t_near_x = t1_x if t1_x < t2_x else t2_x
    t_far_x = t2_x if t2_x > t1_x else t1_x
    t1_y = (min_y - origin_y) * inv_dir_y
    t2_y = (max_y - origin_y) * inv_dir_y
    t_near_y = t1_y if t1_y < t2_y else t2_y
    t_far_y = t2_y if t2_y > t1_y else t1_y
    t1_z = (min_z - origin_z) * inv_dir_z
    t2_z = (max_z - origin_z) * inv_dir_z
    t_near_z = t1_z if t1_z < t2_z else t2_z
    t_far_z = t2_z if t2_z > t1_z else t1_z
    # Find the intersection interval [t_near, t_far] by taking the max of near distances and min of far distances
    t_near = t_near_x
    if t_near_y > t_near:
        t_near = t_near_y
    if t_near_z > t_near:
        t_near = t_near_z
    t_far = t_far_x
    if t_far_y < t_far:
        t_far = t_far_y
    if t_far_z < t_far:
        t_far = t_far_z
    # If the intervals do not overlap or are behind the origin, there's no hit
    if t_near > t_far or t_far < 0.0:
        return math.inf
    # Return the nearest positive intersection distance
    return t_near if t_near >= 0.0 else t_far


@cuda.jit(device=True)
def normal_cube_device(
    point_x, point_y, point_z, center_x, center_y, center_z, size_x, size_y, size_z
):
    # Determine which face of the cube was hit by comparing the point to the cube's center ± half-size
    half_x = size_x * 0.5
    half_y = size_y * 0.5
    half_z = size_z * 0.5
    # Difference between hit point and cube center
    dx = point_x - center_x
    dy = point_y - center_y
    dz = point_z - center_z
    # Compare against ±half extent on each axis with tolerance EPSILON
    if abs(dx - half_x) < EPSILON:
        return 1.0, 0.0, 0.0  # hit right face (+X)
    if abs(dx + half_x) < EPSILON:
        return -1.0, 0.0, 0.0  # hit left face (-X)
    if abs(dy - half_y) < EPSILON:
        return 0.0, 1.0, 0.0  # hit top face (+Y)
    if abs(dy + half_y) < EPSILON:
        return 0.0, -1.0, 0.0  # hit bottom face (-Y)
    if abs(dz - half_z) < EPSILON:
        return 0.0, 0.0, 1.0  # hit front face (+Z)
    if abs(dz + half_z) < EPSILON:
        return 0.0, 0.0, -1.0  # hit back face (-Z)
    # Fallback (should not happen if point is on surface): return a zero normal
    return 0.0, 0.0, 0.0
