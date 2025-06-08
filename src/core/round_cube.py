import math
from numba import cuda
from . import EPSILON

SIDE_LENGTH = 24.0
ROUND_RADIUS = 2.0
HALF_EXTENT = SIDE_LENGTH * 0.5 - ROUND_RADIUS


@cuda.jit(device=True)
def intersect_round_cube_device(
    origin_x,
    origin_y,
    origin_z,
    dir_x,
    dir_y,
    dir_z,
    center_x,
    center_y,
    center_z,
    rot_mat,
) -> float:
    ox = origin_x - center_x
    oy = origin_y - center_y
    oz = origin_z - center_z

    lx = rot_mat[0, 0] * ox + rot_mat[1, 0] * oy + rot_mat[2, 0] * oz
    ly = rot_mat[0, 1] * ox + rot_mat[1, 1] * oy + rot_mat[2, 1] * oz
    lz = rot_mat[0, 2] * ox + rot_mat[1, 2] * oy + rot_mat[2, 2] * oz

    dx = rot_mat[0, 0] * dir_x + rot_mat[1, 0] * dir_y + rot_mat[2, 0] * dir_z
    dy = rot_mat[0, 1] * dir_x + rot_mat[1, 1] * dir_y + rot_mat[2, 1] * dir_z
    dz = rot_mat[0, 2] * dir_x + rot_mat[1, 2] * dir_y + rot_mat[2, 2] * dir_z

    best_t = math.inf
    r = ROUND_RADIUS
    hx = HALF_EXTENT
    hy = HALF_EXTENT
    hz = HALF_EXTENT

    if abs(dx) > EPSILON:
        for side in (hx, -hx):
            t = (side - lx) / dx
            if t > EPSILON:
                y_hit = ly + t * dy
                z_hit = lz + t * dz
                if abs(y_hit) <= hy and abs(z_hit) <= hz and t < best_t:
                    best_t = t
    if abs(dy) > EPSILON:
        for side in (hy, -hy):
            t = (side - ly) / dy
            if t > EPSILON:
                x_hit = lx + t * dx
                z_hit = lz + t * dz
                if abs(x_hit) <= hx and abs(z_hit) <= hz and t < best_t:
                    best_t = t
    if abs(dz) > EPSILON:
        for side in (hz, -hz):
            t = (side - lz) / dz
            if t > EPSILON:
                x_hit = lx + t * dx
                y_hit = ly + t * dy
                if abs(x_hit) <= hx and abs(y_hit) <= hy and t < best_t:
                    best_t = t

    A = dy * dy + dz * dz
    if A > EPSILON:
        for sy in (hy, -hy):
            for sz in (hz, -hz):
                oy_s = ly - sy
                oz_s = lz - sz
                B = 2.0 * (oy_s * dy + oz_s * dz)
                C = oy_s * oy_s + oz_s * oz_s - r * r
                disc = B * B - 4.0 * A * C
                if disc >= 0.0:
                    sd = math.sqrt(disc)
                    for root in (-sd, sd):
                        t_candidate = (-B + root) / (2.0 * A)
                        if t_candidate > EPSILON:
                            x_hit = lx + t_candidate * dx
                            if abs(x_hit) <= hx and t_candidate < best_t:
                                best_t = t_candidate

    A = dx * dx + dz * dz
    if A > EPSILON:
        for sx in (hx, -hx):
            for sz in (hz, -hz):
                ox_s = lx - sx
                oz_s = lz - sz
                B = 2.0 * (ox_s * dx + oz_s * dz)
                C = ox_s * ox_s + oz_s * oz_s - r * r
                disc = B * B - 4.0 * A * C
                if disc >= 0.0:
                    sd = math.sqrt(disc)
                    for root in (-sd, sd):
                        t_candidate = (-B + root) / (2.0 * A)
                        if t_candidate > EPSILON:
                            y_hit = ly + t_candidate * dy
                            if abs(y_hit) <= hy and t_candidate < best_t:
                                best_t = t_candidate

    A = dx * dx + dy * dy
    if A > EPSILON:
        for sx in (hx, -hx):
            for sy in (hy, -hy):
                ox_s = lx - sx
                oy_s = ly - sy
                B = 2.0 * (ox_s * dx + oy_s * dy)
                C = ox_s * ox_s + oy_s * oy_s - r * r
                disc = B * B - 4.0 * A * C
                if disc >= 0.0:
                    sd = math.sqrt(disc)
                    for root in (-sd, sd):
                        t_candidate = (-B + root) / (2.0 * A)
                        if t_candidate > EPSILON:
                            z_hit = lz + t_candidate * dz
                            if abs(z_hit) <= hz and t_candidate < best_t:
                                best_t = t_candidate

    for sx in (hx, -hx):
        for sy in (hy, -hy):
            for sz in (hz, -hz):
                ox_s = lx - sx
                oy_s = ly - sy
                oz_s = lz - sz
                B = ox_s * dx + oy_s * dy + oz_s * dz
                C = ox_s * ox_s + oy_s * oy_s + oz_s * oz_s - r * r
                disc = B * B - C
                if disc >= 0.0:
                    sd = math.sqrt(disc)
                    for root in (-sd, sd):
                        t_candidate = -B + root
                        if t_candidate > EPSILON and t_candidate < best_t:
                            best_t = t_candidate

    return best_t


@cuda.jit(device=True)
def normal_round_cube_device(
    hit_x, hit_y, hit_z, center_x, center_y, center_z, rot_mat
):
    ox = hit_x - center_x
    oy = hit_y - center_y
    oz = hit_z - center_z
    lx = rot_mat[0, 0] * ox + rot_mat[1, 0] * oy + rot_mat[2, 0] * oz
    ly = rot_mat[0, 1] * ox + rot_mat[1, 1] * oy + rot_mat[2, 1] * oz
    lz = rot_mat[0, 2] * ox + rot_mat[1, 2] * oy + rot_mat[2, 2] * oz

    dx = abs(lx) - HALF_EXTENT
    dy = abs(ly) - HALF_EXTENT
    dz = abs(lz) - HALF_EXTENT
    qx = dx if dx > 0.0 else 0.0
    qy = dy if dy > 0.0 else 0.0
    qz = dz if dz > 0.0 else 0.0
    length = math.sqrt(qx * qx + qy * qy + qz * qz)

    if length > EPSILON:
        inv = 1.0 / length
        nx_local = (qx * inv) * (1.0 if lx >= 0.0 else -1.0)
        ny_local = (qy * inv) * (1.0 if ly >= 0.0 else -1.0)
        nz_local = (qz * inv) * (1.0 if lz >= 0.0 else -1.0)
    else:
        ax = HALF_EXTENT - abs(lx)
        ay = HALF_EXTENT - abs(ly)
        az = HALF_EXTENT - abs(lz)
        if ax < ay and ax < az:
            nx_local, ny_local, nz_local = (1.0 if lx >= 0 else -1.0), 0.0, 0.0
        elif ay < az:
            nx_local, ny_local, nz_local = 0.0, (1.0 if ly >= 0 else -1.0), 0.0
        else:
            nx_local, ny_local, nz_local = 0.0, 0.0, (1.0 if lz >= 0 else -1.0)

    nx = rot_mat[0, 0] * nx_local + rot_mat[0, 1] * ny_local + rot_mat[0, 2] * nz_local
    ny = rot_mat[1, 0] * nx_local + rot_mat[1, 1] * ny_local + rot_mat[1, 2] * nz_local
    nz = rot_mat[2, 0] * nx_local + rot_mat[2, 1] * ny_local + rot_mat[2, 2] * nz_local

    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm_len > EPSILON:
        inv = 1.0 / norm_len
        return nx * inv, ny * inv, nz * inv

    return 0.0, 1.0, 0.0
