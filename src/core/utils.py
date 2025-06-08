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


@cuda.jit(device=True)
def sample_hemisphere(norm_x, norm_y, norm_z, states, idx):
    # Compute tangent (T) and bitangent (B) for normal (N)
    if abs(norm_y) < 0.999:
        up_x, up_y, up_z = 0.0, 1.0, 0.0  # reference up vector
    else:
        up_x, up_y, up_z = 1.0, 0.0, 0.0
    # Tangent = up × N
    tang_x = up_y * norm_z - up_z * norm_y
    tang_y = up_z * norm_x - up_x * norm_z
    tang_z = up_x * norm_y - up_y * norm_x
    tang_x, tang_y, tang_z = normalize3(tang_x, tang_y, tang_z)
    # Bitangent = N × T
    bit_x = norm_y * tang_z - norm_z * tang_y
    bit_y = norm_z * tang_x - norm_x * tang_z
    bit_z = norm_x * tang_y - norm_y * tang_x
    # Random angles for cosine-weighted hemisphere
    u = cuda.random.xoroshiro128p_uniform_float32(states, idx)
    v = cuda.random.xoroshiro128p_uniform_float32(states, idx)
    phi = 2 * math.pi * u
    cos_theta = math.sqrt(1 - v)
    sin_theta = math.sqrt(v)
    # Local sample direction (hemisphere around local Y-axis)
    local_x = math.cos(phi) * sin_theta
    local_y = cos_theta
    local_z = math.sin(phi) * sin_theta
    # Transform local direction to world coordinates
    new_dir_x = tang_x * local_x + norm_x * local_y + bit_x * local_z
    new_dir_y = tang_y * local_x + norm_y * local_y + bit_y * local_z
    new_dir_z = tang_z * local_x + norm_z * local_y + bit_z * local_z
    new_dir_x, new_dir_y, new_dir_z = normalize3(new_dir_x, new_dir_y, new_dir_z)
    return new_dir_x, new_dir_y, new_dir_z


@cuda.jit(device=True)
def gamma_correct(val, gamma=2.2):
    val = min(max(val, 0.0), 1.0)
    return math.pow(val, 1.0 / gamma)
