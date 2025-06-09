import math
from numba import cuda
from . import (
    LIGHT_LEN_X,
    LIGHT_LEN_Z,
    LIGHT_Y,
    SHAPE_CUBE,
    SHAPE_SPHERE,
    SHAPE_CYLINDER,
    SHAPE_EGG,
    SHAPE_ROUND_CUBE,
    SHAPE_STICKER,
)


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


@cuda.jit(device=True)
def sample_reflection(
    norm_x, norm_y, norm_z, in_dir_x, in_dir_y, in_dir_z, shape_type, states, idx
):
    if abs(norm_y) < 0.999:
        up_x, up_y, up_z = 0.0, 1.0, 0.0
    else:
        up_x, up_y, up_z = 1.0, 0.0, 0.0
    tang_x = up_y * norm_z - up_z * norm_y
    tang_y = up_z * norm_x - up_x * norm_z
    tang_z = up_x * norm_y - up_y * norm_x
    tang_x, tang_y, tang_z = normalize3(tang_x, tang_y, tang_z)
    bit_x = norm_y * tang_z - norm_z * tang_y
    bit_y = norm_z * tang_x - norm_x * tang_z
    bit_z = norm_x * tang_y - norm_y * tang_x

    u = cuda.random.xoroshiro128p_uniform_float32(states, idx)
    v = cuda.random.xoroshiro128p_uniform_float32(states, idx)
    phi = 2 * math.pi * u
    cos_theta = math.sqrt(1 - v)
    sin_theta = math.sqrt(v)
    local_x = math.cos(phi) * sin_theta
    local_y = cos_theta
    local_z = math.sin(phi) * sin_theta

    diff_x = tang_x * local_x + norm_x * local_y + bit_x * local_z
    diff_y = tang_y * local_x + norm_y * local_y + bit_y * local_z
    diff_z = tang_z * local_x + norm_z * local_y + bit_z * local_z
    diff_x, diff_y, diff_z = normalize3(diff_x, diff_y, diff_z)

    if shape_type not in (SHAPE_EGG, SHAPE_STICKER):
        return diff_x, diff_y, diff_z

    dot_nl = dot3(norm_x, norm_y, norm_z, -in_dir_x, -in_dir_y, -in_dir_z)
    reflect_x = 2 * dot_nl * norm_x + in_dir_x
    reflect_y = 2 * dot_nl * norm_y + in_dir_y
    reflect_z = 2 * dot_nl * norm_z + in_dir_z
    reflect_x, reflect_y, reflect_z = normalize3(reflect_x, reflect_y, reflect_z)

    blend = 0.6 if shape_type == SHAPE_EGG else 0.4

    out_x = reflect_x * (1 - blend) + diff_x * blend
    out_y = reflect_y * (1 - blend) + diff_y * blend
    out_z = reflect_z * (1 - blend) + diff_z * blend
    return out_x, out_y, out_z


@cuda.jit(device=True)
def hits_light_plane(ox, oy, oz, dx, dy, dz):
    if dy == 0.0:
        return False

    t = (LIGHT_Y - oy) / dy
    if t <= 0.0:
        return False

    px = ox + t * dx
    pz = oz + t * dz

    if (
        -LIGHT_LEN_X / 2 <= px <= LIGHT_LEN_X / 2
        and -LIGHT_LEN_Z / 2 <= pz <= LIGHT_LEN_Z / 2
    ):
        return True
    return False


@cuda.jit(device=True)
def gamma_correct(val):
    val = min(max(val, 0.0), 1.0)
    return math.pow(val, 1.0 / 2.2)
