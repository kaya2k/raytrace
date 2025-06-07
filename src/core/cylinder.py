import math
from numba import cuda
from . import EPSILON


@cuda.jit(device=True)
def intersect_cylinder_device(
    origin_x,
    origin_y,
    origin_z,
    dir_x,
    dir_y,
    dir_z,
    center_x,
    center_y,
    center_z,
    height,
    radius,
) -> float:
    ox = origin_x - center_x
    oy = origin_y - center_y
    oz = origin_z - center_z
    dx = dir_x
    dy = dir_y
    dz = dir_z

    half_h = height * 0.5
    t_min = math.inf

    # side
    a = dx * dx + dz * dz
    b = ox * dx + oz * dz
    c = ox * ox + oz * oz - radius * radius
    disc = b * b - a * c
    if disc >= 0.0 and abs(a) > EPSILON:
        sqrt_disc = math.sqrt(disc)
        t0 = (-b - sqrt_disc) / a
        t1 = (-b + sqrt_disc) / a
        if t0 > EPSILON:
            y0 = oy + t0 * dy
            if -half_h <= y0 <= half_h:
                t_min = t0
        if t_min == math.inf and t1 > EPSILON:
            y1 = oy + t1 * dy
            if -half_h <= y1 <= half_h:
                t_min = t1

    # top cap
    if abs(dy) > EPSILON:
        t_cap = (half_h - oy) / dy
        if t_cap > EPSILON:
            x_cap = ox + t_cap * dx
            z_cap = oz + t_cap * dz
            if x_cap * x_cap + z_cap * z_cap <= radius * radius:
                if t_cap < t_min:
                    t_min = t_cap

    # bottom cap
    if abs(dy) > EPSILON:
        t_cap = (-half_h - oy) / dy
        if t_cap > EPSILON:
            x_cap = ox + t_cap * dx
            z_cap = oz + t_cap * dz
            if x_cap * x_cap + z_cap * z_cap <= radius * radius:
                if t_cap < t_min:
                    t_min = t_cap

    return t_min if t_min < math.inf else math.inf


@cuda.jit(device=True)
def normal_cylinder_device(
    point_x, point_y, point_z, center_x, center_y, center_z, height, radius
):
    px = point_x - center_x
    py = point_y - center_y
    pz = point_z - center_z
    half_h = height * 0.5

    # top cap
    if abs(py - half_h) < EPSILON and (px * px + pz * pz) <= radius * radius:
        return 0.0, 1.0, 0.0

    # bottom cap
    if abs(py + half_h) < EPSILON and (px * px + pz * pz) <= radius * radius:
        return 0.0, -1.0, 0.0

    # side normal
    length = math.sqrt(px * px + pz * pz)
    if length > EPSILON:
        inv = 1.0 / length
        return px * inv, 0.0, pz * inv

    # 예외 상태: 0 벡터 반환
    return 0.0, 0.0, 0.0
