import math
from numba import cuda
from . import (
    EPSILON,
    SHAPE_CUBE,
    SHAPE_SPHERE,
    SHAPE_CYLINDER,
    SHAPE_EGG,
    SHAPE_ROUND_CUBE,
    SHAPE_STICKER,
    EGG_R,
    EGG_G,
    EGG_B,
    ROUND_CUBE_R,
    ROUND_CUBE_G,
    ROUND_CUBE_B,
)
from .cube import intersect_cube_device, normal_cube_device
from .sphere import intersect_sphere_device, normal_sphere_device
from .cylinder import intersect_cylinder_device, normal_cylinder_device
from .egg import intersect_egg_device, normal_egg_device
from .round_cube import intersect_round_cube_device, normal_round_cube_device
from .sticker import intersect_sticker_device, normal_sticker_device
from .utils import dot3, normalize3

LIGHT_SAMPLES_PER_CELL = 4
LIGHT_Y = 105.0
LIGHT_LEN_X = 58.0
LIGHT_LEN_Z = 44.0
LIGHT_NX = 6
LIGHT_NZ = int(LIGHT_NX * LIGHT_LEN_Z / LIGHT_LEN_X)
CELL_SIZE_X = LIGHT_LEN_X / LIGHT_NX
CELL_SIZE_Z = LIGHT_LEN_Z / LIGHT_NZ

AMBI = 0.30
DIFF_C = 0.80
SPEC_C = 0.10
SPEC_K = 16.0
EGG_SPEC_C = 0.30
EGG_SPEC_K = 256

MAX_DEPTH = 2
REFLECT_EGG = 1.0
REFLECT_STICKER = 1.0


@cuda.jit(device=True)
def check_in_shadow_device(
    point_x,
    point_y,
    point_z,
    to_light_x,
    to_light_y,
    to_light_z,
    norm_x,
    norm_y,
    norm_z,
    light_distance,
    cube_min_bounds,
    cube_max_bounds,
    sphere_centers,
    sphere_radii,
    cylinder_centers,
    cylinder_heights,
    cylinder_radii,
    round_cube_centers,
    round_cube_rotations,
    sticker_centers,
    sticker_rotations,
    sticker_colors,
    n_cubes,
    n_spheres,
    n_cylinders,
    n_round_cubes,
    n_stickers,
) -> bool:
    # Offset the start point a bit along the light ray to avoid self-hit
    origin_x = point_x + norm_x * (EPSILON * 100.0)
    origin_y = point_y + norm_y * (EPSILON * 100.0)
    origin_z = point_z + norm_z * (EPSILON * 100.0)
    # Check intersection with every cube
    for i in range(n_cubes):
        d = intersect_cube_device(
            origin_x,
            origin_y,
            origin_z,
            to_light_x,
            to_light_y,
            to_light_z,
            cube_min_bounds[i, 0],
            cube_min_bounds[i, 1],
            cube_min_bounds[i, 2],
            cube_max_bounds[i, 0],
            cube_max_bounds[i, 1],
            cube_max_bounds[i, 2],
        )
        if d < light_distance:
            return True  # something blocks the light before it reaches the point
    # Check intersection with every sphere
    for i in range(n_spheres):
        d = intersect_sphere_device(
            origin_x,
            origin_y,
            origin_z,
            to_light_x,
            to_light_y,
            to_light_z,
            sphere_centers[i, 0],
            sphere_centers[i, 1],
            sphere_centers[i, 2],
            sphere_radii[i],
        )
        if d < light_distance:
            return True
    # Check intersection with every cylinder
    for i in range(n_cylinders):
        d = intersect_cylinder_device(
            origin_x,
            origin_y,
            origin_z,
            to_light_x,
            to_light_y,
            to_light_z,
            cylinder_centers[i, 0],
            cylinder_centers[i, 1],
            cylinder_centers[i, 2],
            cylinder_heights[i],
            cylinder_radii[i],
        )
        if d < light_distance:
            return True
    # Check intersection with the egg shape
    d = intersect_egg_device(
        origin_x,
        origin_y,
        origin_z,
        to_light_x,
        to_light_y,
        to_light_z,
    )
    if d < light_distance:
        return True
    # Check intersection with every round cube
    for i in range(n_round_cubes):
        d = intersect_round_cube_device(
            origin_x,
            origin_y,
            origin_z,
            to_light_x,
            to_light_y,
            to_light_z,
            round_cube_centers[i, 0],
            round_cube_centers[i, 1],
            round_cube_centers[i, 2],
            round_cube_rotations[i],
        )
        if d < light_distance:
            return True
    # Check intersection with every sticker
    for i in range(n_stickers):
        d = intersect_sticker_device(
            origin_x,
            origin_y,
            origin_z,
            to_light_x,
            to_light_y,
            to_light_z,
            sticker_centers[i, 0],
            sticker_centers[i, 1],
            sticker_centers[i, 2],
            sticker_rotations[i],
        )
        if d < light_distance:
            return True

    return False  # no object obstructs the light


@cuda.jit(device=True)
def compute_color_device(
    to_orig_x,
    to_orig_y,
    to_orig_z,
    point_x,
    point_y,
    point_z,
    norm_x,
    norm_y,
    norm_z,
    shape_col_r,
    shape_col_g,
    shape_col_b,
    cube_min_bounds,
    cube_max_bounds,
    sphere_centers,
    sphere_radii,
    cylinder_centers,
    cylinder_heights,
    cylinder_radii,
    round_cube_centers,
    round_cube_rotations,
    sticker_centers,
    sticker_rotations,
    sticker_colors,
    states,
    idx,
    n_cubes,
    n_spheres,
    n_cylinders,
    n_round_cubes,
    n_stickers,
    shape_type,
):
    # Accumulate lighting contributions
    color_r = 0.0
    color_g = 0.0
    color_b = 0.0
    # For each cell of the area light
    for i in range(LIGHT_NX):
        # Compute the X coordinate range of this cell on the light
        x_min = i * CELL_SIZE_X - 0.5 * LIGHT_LEN_X
        for j in range(LIGHT_NZ):
            # Z coordinate range of this cell
            z_min = j * CELL_SIZE_Z - 0.5 * LIGHT_LEN_Z
            # Take multiple random samples within each cell
            for _ in range(LIGHT_SAMPLES_PER_CELL):
                # Random offsets (u,v) in [0,1) for this cell
                u = cuda.random.xoroshiro128p_uniform_float32(states, idx)
                v = cuda.random.xoroshiro128p_uniform_float32(states, idx)
                # Sample point on the light area
                sample_x = x_min + u * CELL_SIZE_X
                sample_z = z_min + v * CELL_SIZE_Z
                # Vector from hit point to the light sample
                to_light_x = sample_x - point_x
                to_light_y = LIGHT_Y - point_y
                to_light_z = sample_z - point_z
                # Distance to the light sample
                dist = math.sqrt(
                    to_light_x * to_light_x
                    + to_light_y * to_light_y
                    + to_light_z * to_light_z
                )
                if dist <= 0.0:
                    continue  # point is at the light (degenerate case)
                # Normalize the to_light direction
                to_light_x /= dist
                to_light_y /= dist
                to_light_z /= dist
                light_distance = dist  # distance to light
                # Shadow check: is the light visible from the point?
                if not check_in_shadow_device(
                    point_x,
                    point_y,
                    point_z,
                    to_light_x,
                    to_light_y,
                    to_light_z,
                    norm_x,
                    norm_y,
                    norm_z,
                    light_distance,
                    cube_min_bounds,
                    cube_max_bounds,
                    sphere_centers,
                    sphere_radii,
                    cylinder_centers,
                    cylinder_heights,
                    cylinder_radii,
                    round_cube_centers,
                    round_cube_rotations,
                    sticker_centers,
                    sticker_rotations,
                    sticker_colors,
                    n_cubes,
                    n_spheres,
                    n_cylinders,
                    n_round_cubes,
                    n_stickers,
                ):
                    # Light is not blocked – add diffuse and specular contribution
                    # Diffuse component (Lambertian): shape_color * (N·L) * coefficient
                    # dot(N,L):
                    diffuse_factor = dot3(
                        norm_x, norm_y, norm_z, to_light_x, to_light_y, to_light_z
                    )
                    if diffuse_factor < 0.0:
                        diffuse_factor = 0.0
                    color_r += DIFF_C * shape_col_r * diffuse_factor
                    color_g += DIFF_C * shape_col_g * diffuse_factor
                    color_b += DIFF_C * shape_col_b * diffuse_factor
                    # Specular component (Blinn-Phong):
                    # Half vector = normalized (to_light + to_origin)
                    half_x = to_light_x + to_orig_x
                    half_y = to_light_y + to_orig_y
                    half_z = to_light_z + to_orig_z
                    half_x, half_y, half_z = normalize3(half_x, half_y, half_z)
                    # dot(N, H):
                    let = dot3(norm_x, norm_y, norm_z, half_x, half_y, half_z)
                    if let < 0.0:
                        let = 0.0
                    # Specular intensity
                    spec = EGG_SPEC_C if shape_type == SHAPE_EGG else SPEC_C
                    spec *= let ** (EGG_SPEC_K if shape_type == SHAPE_EGG else SPEC_K)
                    color_r += spec
                    color_g += spec
                    color_b += spec
    # Average the accumulated light contributions over all samples
    total_samples = LIGHT_SAMPLES_PER_CELL * LIGHT_NX * LIGHT_NZ
    color_r /= total_samples
    color_g /= total_samples
    color_b /= total_samples
    # Add ambient term (ambient light uniformly adds a base color)
    color_r += AMBI * shape_col_r
    color_g += AMBI * shape_col_g
    color_b += AMBI * shape_col_b
    return color_r, color_g, color_b


@cuda.jit(device=True)
def trace_ray_device(
    origin_x,
    origin_y,
    origin_z,
    dir_x,
    dir_y,
    dir_z,
    cube_centers,
    cube_sizes,
    cube_colors,
    cube_min_bounds,
    cube_max_bounds,
    sphere_centers,
    sphere_radii,
    sphere_colors,
    cylinder_centers,
    cylinder_heights,
    cylinder_radii,
    cylinder_colors,
    round_cube_centers,
    round_cube_rotations,
    sticker_centers,
    sticker_rotations,
    sticker_colors,
    states,
    idx,
    n_cubes,
    n_spheres,
    n_cylinders,
    n_round_cubes,
    n_stickers,
):
    color_r = 0.0
    color_g = 0.0
    color_b = 0.0
    attenuation = 1.0

    ox = origin_x
    oy = origin_y
    oz = origin_z
    dx = dir_x
    dy = dir_y
    dz = dir_z

    for _ in range(MAX_DEPTH):
        min_t = math.inf
        min_shape = -1
        min_idx = -1

        # cubes
        for i in range(n_cubes):
            t = intersect_cube_device(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                cube_min_bounds[i, 0],
                cube_min_bounds[i, 1],
                cube_min_bounds[i, 2],
                cube_max_bounds[i, 0],
                cube_max_bounds[i, 1],
                cube_max_bounds[i, 2],
            )
            if t < min_t:
                min_t = t
                min_shape = SHAPE_CUBE
                min_idx = i
        # spheres
        for i in range(n_spheres):
            t = intersect_sphere_device(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                sphere_centers[i, 0],
                sphere_centers[i, 1],
                sphere_centers[i, 2],
                sphere_radii[i],
            )
            if t < min_t:
                min_t = t
                min_shape = SHAPE_SPHERE
                min_idx = i
        # cylinders
        for i in range(n_cylinders):
            t = intersect_cylinder_device(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                cylinder_centers[i, 0],
                cylinder_centers[i, 1],
                cylinder_centers[i, 2],
                cylinder_heights[i],
                cylinder_radii[i],
            )
            if t < min_t:
                min_t = t
                min_shape = SHAPE_CYLINDER
                min_idx = i
        # egg
        t = intersect_egg_device(ox, oy, oz, dx, dy, dz)
        if t < min_t:
            min_t = t
            min_shape = SHAPE_EGG
        # round cubes
        for i in range(n_round_cubes):
            t = intersect_round_cube_device(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                round_cube_centers[i, 0],
                round_cube_centers[i, 1],
                round_cube_centers[i, 2],
                round_cube_rotations[i],
            )
            if t < min_t:
                min_t = t
                min_shape = SHAPE_ROUND_CUBE
                min_idx = i

        # stickers
        for i in range(n_stickers):
            t = intersect_sticker_device(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                sticker_centers[i, 0],
                sticker_centers[i, 1],
                sticker_centers[i, 2],
                sticker_rotations[i],
            )
            if t < min_t:
                min_t = t
                min_shape = SHAPE_STICKER
                min_idx = i

        if min_shape == -1:
            if dz < 0.0:  # light source
                color_r += attenuation * 1.0
                color_g += attenuation * 1.0
                color_b += attenuation * 1.0
            break

        # hit point and normal
        hit_x = ox + min_t * dx
        hit_y = oy + min_t * dy
        hit_z = oz + min_t * dz

        if min_shape == SHAPE_CUBE:
            norm_x, norm_y, norm_z = normal_cube_device(
                hit_x,
                hit_y,
                hit_z,
                cube_centers[min_idx, 0],
                cube_centers[min_idx, 1],
                cube_centers[min_idx, 2],
                cube_sizes[min_idx, 0],
                cube_sizes[min_idx, 1],
                cube_sizes[min_idx, 2],
            )
            shape_r = cube_colors[min_idx, 0]
            shape_g = cube_colors[min_idx, 1]
            shape_b = cube_colors[min_idx, 2]
        elif min_shape == SHAPE_SPHERE:
            norm_x, norm_y, norm_z = normal_sphere_device(
                hit_x,
                hit_y,
                hit_z,
                sphere_centers[min_idx, 0],
                sphere_centers[min_idx, 1],
                sphere_centers[min_idx, 2],
            )
            shape_r = sphere_colors[min_idx, 0]
            shape_g = sphere_colors[min_idx, 1]
            shape_b = sphere_colors[min_idx, 2]
        elif min_shape == SHAPE_CYLINDER:
            norm_x, norm_y, norm_z = normal_cylinder_device(
                hit_x,
                hit_y,
                hit_z,
                cylinder_centers[min_idx, 0],
                cylinder_centers[min_idx, 1],
                cylinder_centers[min_idx, 2],
                cylinder_heights[min_idx],
                cylinder_radii[min_idx],
            )
            shape_r = cylinder_colors[min_idx, 0]
            shape_g = cylinder_colors[min_idx, 1]
            shape_b = cylinder_colors[min_idx, 2]
        elif min_shape == SHAPE_EGG:
            norm_x, norm_y, norm_z = normal_egg_device(hit_x, hit_y, hit_z)
            shape_r, shape_g, shape_b = EGG_R, EGG_G, EGG_B
        elif min_shape == SHAPE_ROUND_CUBE:
            norm_x, norm_y, norm_z = normal_round_cube_device(
                hit_x,
                hit_y,
                hit_z,
                round_cube_centers[min_idx, 0],
                round_cube_centers[min_idx, 1],
                round_cube_centers[min_idx, 2],
                round_cube_rotations[min_idx],
            )
            shape_r, shape_g, shape_b = ROUND_CUBE_R, ROUND_CUBE_G, ROUND_CUBE_B
        else:
            norm_x, norm_y, norm_z = normal_sticker_device(sticker_rotations[min_idx])
            shape_r, shape_g, shape_b = (
                sticker_colors[min_idx, 0],
                sticker_colors[min_idx, 1],
                sticker_colors[min_idx, 2],
            )

        # direct lighting
        dr, dg, db = compute_color_device(
            -dx,
            -dy,
            -dz,
            hit_x,
            hit_y,
            hit_z,
            norm_x,
            norm_y,
            norm_z,
            shape_r,
            shape_g,
            shape_b,
            cube_min_bounds,
            cube_max_bounds,
            sphere_centers,
            sphere_radii,
            cylinder_centers,
            cylinder_heights,
            cylinder_radii,
            round_cube_centers,
            round_cube_rotations,
            sticker_centers,
            sticker_rotations,
            sticker_colors,
            states,
            idx,
            n_cubes,
            n_spheres,
            n_cylinders,
            n_round_cubes,
            n_stickers,
            min_shape,
        )
        color_r += attenuation * dr
        color_g += attenuation * dg
        color_b += attenuation * db

        # prepare reflection
        if min_shape == SHAPE_STICKER:
            reflect_f = REFLECT_STICKER
        elif min_shape == SHAPE_EGG:
            reflect_f = REFLECT_EGG
        else:
            reflect_f = 1.0

        attenuation *= reflect_f
        # reflection direction
        dot_dn = dot3(dx, dy, dz, norm_x, norm_y, norm_z)
        rx = dx - 2.0 * dot_dn * norm_x
        ry = dy - 2.0 * dot_dn * norm_y
        rz = dz - 2.0 * dot_dn * norm_z
        # offset origin and normalize direction
        ox = hit_x + norm_x * (EPSILON * 100.0)
        oy = hit_y + norm_y * (EPSILON * 100.0)
        oz = hit_z + norm_z * (EPSILON * 100.0)
        rx, ry, rz = normalize3(rx, ry, rz)
        dx, dy, dz = rx, ry, rz

    return color_r, color_g, color_b
