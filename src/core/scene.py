import math
from numba import cuda
from . import EPSILON, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_CYLINDER
from .cube import intersect_cube_device, normal_cube_device
from .sphere import intersect_sphere_device, normal_sphere_device
from .cylinder import intersect_cylinder_device, normal_cylinder_device
from .utils import dot3, normalize3

# Constants for the area light and Phong illumination
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
SPEC_C = 0.20
SPEC_K = 32


@cuda.jit(device=True)
def check_in_shadow_device(
    point_x,
    point_y,
    point_z,
    to_light_x,
    to_light_y,
    to_light_z,
    light_distance,
    cube_min_bounds,
    cube_max_bounds,
    sphere_centers,
    sphere_radii,
    cylinder_centers,
    cylinder_heights,
    cylinder_radii,
    n_cubes,
    n_spheres,
    n_cylinders,
) -> bool:
    # Offset the start point a bit along the light ray to avoid self-hit
    origin_x = point_x + to_light_x * (EPSILON * 10.0)
    origin_y = point_y + to_light_y * (EPSILON * 10.0)
    origin_z = point_z + to_light_z * (EPSILON * 10.0)
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
    cylinder_colors,
    states,
    idx,
    n_cubes,
    n_spheres,
    n_cylinders,
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
                # Sample point on the light area (rectangle centered at origin in X-Z plane at Y = LIGHT_Y)
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
                    light_distance,
                    cube_min_bounds,
                    cube_max_bounds,
                    sphere_centers,
                    sphere_radii,
                    cylinder_centers,
                    cylinder_heights,
                    cylinder_radii,
                    n_cubes,
                    n_spheres,
                    n_cylinders,
                ):
                    # Light is not blocked – add diffuse and specular contribution
                    # Diffuse component (Lambertian): shape_color * (N·L) * coefficient
                    # dot(N,L):
                    diffuse_factor = dot3(
                        norm_x, norm_y, norm_z, to_light_x, to_light_y, to_light_z
                    )
                    if diffuse_factor < 0.0:
                        diffuse_factor = 0.0  # no negative light contribution
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
                    # We raise max(N·H, 0) to the power SPEC_K. If N·H is negative, we treat it as 0 (no specular).
                    if let < 0.0:
                        let = 0.0
                    # Specular intensity (same for R,G,B, so we use white specular color)
                    spec = SPEC_C * (let**SPEC_K)
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
    states,
    idx,
    n_cubes,
    n_spheres,
    n_cylinders,
):
    # Find the closest intersection of the ray with any object
    min_t = math.inf
    min_shape = -1  # -1 means no hit
    min_idx = -1
    # Test all cubes
    for i in range(n_cubes):
        t = intersect_cube_device(
            origin_x,
            origin_y,
            origin_z,
            dir_x,
            dir_y,
            dir_z,
            cube_min_bounds[i, 0],
            cube_min_bounds[i, 1],
            cube_min_bounds[i, 2],
            cube_max_bounds[i, 0],
            cube_max_bounds[i, 1],
            cube_max_bounds[i, 2],
        )
        if t < min_t:
            min_t = t
            min_idx = i
            min_shape = SHAPE_CUBE
    # Test all spheres
    for i in range(n_spheres):
        t = intersect_sphere_device(
            origin_x,
            origin_y,
            origin_z,
            dir_x,
            dir_y,
            dir_z,
            sphere_centers[i, 0],
            sphere_centers[i, 1],
            sphere_centers[i, 2],
            sphere_radii[i],
        )
        if t < min_t:
            min_t = t
            min_idx = i
            min_shape = SHAPE_SPHERE
    # Test all cylinders
    for i in range(n_cylinders):
        t = intersect_cylinder_device(
            origin_x,
            origin_y,
            origin_z,
            dir_x,
            dir_y,
            dir_z,
            cylinder_centers[i, 0],
            cylinder_centers[i, 1],
            cylinder_centers[i, 2],
            cylinder_heights[i],
            cylinder_radii[i],
        )
        if t < min_t:
            min_t = t
            min_idx = i
            min_shape = SHAPE_CYLINDER
    # If no object was hit, return light color (white)
    if min_shape == -1:
        return 1.0, 1.0, 1.0
    # Compute the exact hit point on the surface
    hit_x = origin_x + min_t * dir_x
    hit_y = origin_y + min_t * dir_y
    hit_z = origin_z + min_t * dir_z
    # Get surface normal and object color at the hit point
    if min_shape == SHAPE_CUBE:
        # Cube hit
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
        shape_col_r = cube_colors[min_idx, 0]
        shape_col_g = cube_colors[min_idx, 1]
        shape_col_b = cube_colors[min_idx, 2]
    elif min_shape == SHAPE_SPHERE:
        # Sphere hit
        norm_x, norm_y, norm_z = normal_sphere_device(
            hit_x,
            hit_y,
            hit_z,
            sphere_centers[min_idx, 0],
            sphere_centers[min_idx, 1],
            sphere_centers[min_idx, 2],
        )
        shape_col_r = sphere_colors[min_idx, 0]
        shape_col_g = sphere_colors[min_idx, 1]
        shape_col_b = sphere_colors[min_idx, 2]
    else:
        # Cylinder hit
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
        shape_col_r = cylinder_colors[min_idx, 0]
        shape_col_g = cylinder_colors[min_idx, 1]
        shape_col_b = cylinder_colors[min_idx, 2]
    # Compute vector from hit point back toward the camera (to_origin = -direction, since direction is normalized)
    to_orig_x = -dir_x
    to_orig_y = -dir_y
    to_orig_z = -dir_z
    # Compute lighting at the hit point using Phong illumination model
    color_r, color_g, color_b = compute_color_device(
        to_orig_x,
        to_orig_y,
        to_orig_z,
        hit_x,
        hit_y,
        hit_z,
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
        cylinder_colors,
        states,
        idx,
        n_cubes,
        n_spheres,
        n_cylinders,
    )
    return color_r, color_g, color_b
