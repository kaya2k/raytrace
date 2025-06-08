import math
from numba import cuda
from . import EPSILON

STICKER_SIDE_LENGTH = 15.8
STICKER_ROUND_RADIUS = 2.0
STICKER_HALF_SIDE = STICKER_SIDE_LENGTH * 0.5
STICKER_OFFSET = 0.1


@cuda.jit(device=True)
def intersect_sticker_device(
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

    lx = rot_mat[0, 0] * ox + rot_mat[0, 1] * oy + rot_mat[0, 2] * oz
    ly = rot_mat[1, 0] * ox + rot_mat[1, 1] * oy + rot_mat[1, 2] * oz
    lz = rot_mat[2, 0] * ox + rot_mat[2, 1] * oy + rot_mat[2, 2] * oz

    dx = rot_mat[0, 0] * dir_x + rot_mat[0, 1] * dir_y + rot_mat[0, 2] * dir_z
    dy = rot_mat[1, 0] * dir_x + rot_mat[1, 1] * dir_y + rot_mat[1, 2] * dir_z
    dz = rot_mat[2, 0] * dir_x + rot_mat[2, 1] * dir_y + rot_mat[2, 2] * dir_z

    if abs(dz) <= EPSILON:
        return math.inf

    t = (STICKER_OFFSET - lz) / dz
    if t <= EPSILON:
        return math.inf

    x = lx + t * dx
    y = ly + t * dy

    qx = abs(x) - STICKER_HALF_SIDE
    qy = abs(y) - STICKER_HALF_SIDE
    if qx <= 0.0 and qy <= 0.0:
        return t

    dx_ = qx if qx > 0.0 else 0.0
    dy_ = qy if qy > 0.0 else 0.0
    if dx_ * dx_ + dy_ * dy_ <= STICKER_ROUND_RADIUS * STICKER_ROUND_RADIUS:
        return t

    return math.inf


@cuda.jit(device=True)
def normal_sticker_device(rot_mat):
    nx = rot_mat[0, 2]
    ny = rot_mat[1, 2]
    nz = rot_mat[2, 2]
    inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * inv, ny * inv, nz * inv
