import math
from numba import cuda
from . import EPSILON

# Implicit egg shape parameters
a = 22.0  # horizontal radius (X,Z)
b_top = 1.4 * a  # top half vertical radius
b_bot = 1.1 * a  # bottom half vertical radius
# Egg center in world coordinates
center_x = 82.0
center_y = -65.0
center_z = -12.0


@cuda.jit(device=True)
def intersect_egg_implicit(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z) -> float:
    # Transform ray origin to egg-local coordinates
    ox = origin_x - center_x
    oy = origin_y - center_y
    oz = origin_z - center_z
    dx = dir_x
    dy = dir_y
    dz = dir_z

    t_min = math.inf

    # Helper to solve quadratic and test region
    def solve_half(b_vert, cond):
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b_vert * b_vert)
        A = dx * dx * inv_a2 + dy * dy * inv_b2 + dz * dz * inv_a2
        B = 2.0 * (ox * dx * inv_a2 + oy * dy * inv_b2 + oz * dz * inv_a2)
        C = ox * ox * inv_a2 + oy * oy * inv_b2 + oz * oz * inv_a2 - 1.0
        disc = B * B - 4.0 * A * C
        if disc < 0.0 or abs(A) < EPSILON:
            return math.inf
        sqrt_d = math.sqrt(disc)
        t0 = (-B - sqrt_d) / (2.0 * A)
        t1 = (-B + sqrt_d) / (2.0 * A)
        t_hit = math.inf
        # Check solutions
        if t0 > EPSILON:
            y0 = oy + t0 * dy
            if cond(y0):
                t_hit = t0
        if t1 > EPSILON:
            y1 = oy + t1 * dy
            if cond(y1) and t1 < t_hit:
                t_hit = t1
        return t_hit

    # Top half (y >= 0)
    t_top = solve_half(b_top, lambda y: y >= 0.0)
    if t_top < t_min:
        t_min = t_top
    # Bottom half (y <= 0)
    t_bot = solve_half(b_bot, lambda y: y <= 0.0)
    if t_bot < t_min:
        t_min = t_bot

    return t_min if t_min < math.inf else math.inf


@cuda.jit(device=True)
def normal_egg_implicit(hit_x, hit_y, hit_z):
    # Transform hit point to egg-local coordinates
    lx = hit_x - center_x
    ly = hit_y - center_y
    lz = hit_z - center_z
    # Determine which half
    if ly >= 0.0:
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b_top * b_top)
    else:
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b_bot * b_bot)
    # Gradient of F(x,y,z) = x^2/a^2 + y^2/b^2 + z^2/a^2 - 1
    nx = 2.0 * lx * inv_a2
    ny = 2.0 * ly * inv_b2
    nz = 2.0 * lz * inv_a2
    # Normalize
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > EPSILON:
        inv = 1.0 / length
        return nx * inv, ny * inv, nz * inv
    return 0.0, 1.0, 0.0
