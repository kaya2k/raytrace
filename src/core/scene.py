import numpy as np
from numba import njit
from . import EPSILON, SHAPE_CUBE, SHAPE_SPHERE
from .utils import dot, normalize
from .cube import intersect_cube, normal_cube
from .sphere import intersect_sphere, normal_sphere


LIGHT_SAMPLES_PER_CELL = 4
LIGHT_Y = 105
LIGHT_LEN_X = 58
LIGHT_LEN_Z = 44
LIGHT_NX = 6
LIGHT_NZ = int(LIGHT_NX * LIGHT_LEN_Z / LIGHT_LEN_X)
CELL_SIZE_X = LIGHT_LEN_X / LIGHT_NX
CELL_SIZE_Z = LIGHT_LEN_Z / LIGHT_NZ

AMBI = 0.30
DIFF_C = 0.8
SPEC_C = 0.2
SPEC_K = 32


@njit(fastmath=True)
def intersect_shapes(
    origin: np.ndarray,
    direction: np.ndarray,
    cube_centers: np.ndarray,
    cube_sizes: np.ndarray,
    cube_colors: np.ndarray,
    cube_min_bounds: np.ndarray,
    cube_max_bounds: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
    sphere_colors: np.ndarray,
) -> tuple:
    min_t = np.inf
    min_shape = -1
    min_i = -1

    # CHECK CUBES
    n_cubes = cube_centers.shape[0]
    for i in range(n_cubes):
        t = intersect_cube(origin, direction, cube_min_bounds[i], cube_max_bounds[i])
        if t < min_t:
            min_t = t
            min_i = i
            min_shape = SHAPE_CUBE

    # CHECK SPHERES
    n_spheres = sphere_centers.shape[0]
    for i in range(n_spheres):
        t = intersect_sphere(origin, direction, sphere_centers[i], sphere_radii[i])
        if t < min_t:
            min_t = t
            min_i = i
            min_shape = SHAPE_SPHERE

    # NO INTERSECTION
    if min_shape == -1:
        return (
            np.array((0.0, 0.0, 0.0)),
            np.array((0.0, 0.0, 0.0)),
            np.array((0.0, 0.0, 0.0)),
            -1,
            -1,
        )

    point = origin + direction * min_t
    if min_shape == SHAPE_CUBE:
        normal = normal_cube(point, cube_centers[min_i], cube_sizes[min_i])
        shape_color = cube_colors[min_i]
    elif min_shape == SHAPE_SPHERE:
        normal = normal_sphere(point, sphere_centers[min_i])
        shape_color = sphere_colors[min_i]
    else:
        assert False

    color = compute_color(
        normalize(origin - point),
        point,
        normal,
        shape_color,
        cube_min_bounds,
        cube_max_bounds,
        sphere_centers,
        sphere_radii,
    )
    return point, normal, color, min_shape, min_i


@njit(fastmath=True)
def compute_color(
    to_origin: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    shape_color: np.ndarray,
    cube_min_bounds: np.ndarray,
    cube_max_bounds: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
) -> np.ndarray:
    color = np.zeros(3)

    for idx in range(LIGHT_NX * LIGHT_NZ):
        x_min = (idx // LIGHT_NZ) * CELL_SIZE_X
        z_min = (idx % LIGHT_NZ) * CELL_SIZE_Z

        for _ in range(LIGHT_SAMPLES_PER_CELL):
            x = x_min + np.random.rand() * CELL_SIZE_X
            z = z_min + np.random.rand() * CELL_SIZE_Z
            light_sample = np.array([x, LIGHT_Y, z])

            to_light = normalize(light_sample - point)
            light_distance = np.linalg.norm(light_sample - point)
            in_shadow = check_in_shadow(
                point,
                to_light,
                float(light_distance),
                cube_min_bounds,
                cube_max_bounds,
                sphere_centers,
                sphere_radii,
            )
            if not in_shadow:
                color += (
                    DIFF_C * shape_color * dot(normal, to_light)
                    + SPEC_C * np.ones(3) * dot(normal, to_origin) ** SPEC_K
                )

    color /= LIGHT_SAMPLES_PER_CELL * LIGHT_NX * LIGHT_NZ
    color += AMBI * shape_color
    return color


@njit(fastmath=True)
def check_in_shadow(
    point: np.ndarray,
    to_light: np.ndarray,
    light_distance: float,
    cube_min_bounds: np.ndarray,
    cube_max_bounds: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
) -> bool:
    origin = point + to_light * EPSILON * 10
    n_cubes = cube_min_bounds.shape[0]
    for i in range(n_cubes):
        d = intersect_cube(origin, to_light, cube_min_bounds[i], cube_max_bounds[i])
        if d < light_distance:
            return True

    n_spheres = sphere_centers.shape[0]
    for i in range(n_spheres):
        d = intersect_sphere(origin, to_light, sphere_centers[i], sphere_radii[i])
        if d < light_distance:
            return True

    return False
