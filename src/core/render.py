import numpy as np
from numba import njit
from .utils import normalize
from .scene import intersect_shapes


IMG_RATIO = 2000 / 1380
IMG_WIDTH = 2000 // 5
IMG_HEIGHT = int(IMG_WIDTH / IMG_RATIO)
IMG_RATIO = IMG_WIDTH / IMG_HEIGHT

CAM_POSITION = np.array((0.0, 0.0, 347.5))
CAM_MAX_X = 210.0
CAM_MAX_Y = CAM_MAX_X / IMG_RATIO

CELL_SIZE_X = CAM_MAX_X * 2 / IMG_WIDTH
CELL_SIZE_Y = CAM_MAX_Y * 2 / IMG_HEIGHT

SS_DIM = 4


@njit(fastmath=True, parallel=True)
def render(
    cube_centers: np.ndarray,
    cube_sizes: np.ndarray,
    cube_colors: np.ndarray,
    cube_min_bounds: np.ndarray,
    cube_max_bounds: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
    sphere_colors: np.ndarray,
) -> np.ndarray:
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    cam_lookat = np.array((0.0, 0.0, 0.0))

    for i, j in np.ndindex(IMG_WIDTH, IMG_HEIGHT):
        if j == 0:
            print(f"Rendering pixel ({i}/{IMG_WIDTH} x {IMG_HEIGHT})")

        x = i * CELL_SIZE_X - CAM_MAX_X
        y = j * CELL_SIZE_Y - CAM_MAX_Y

        color_sum = np.zeros(3, dtype=np.float32)
        for k, l in np.ndindex(SS_DIM, SS_DIM):
            x += (k + 0.5) / SS_DIM * CELL_SIZE_X
            y += (l + 0.5) / SS_DIM * CELL_SIZE_Y

            cam_lookat[0] = x
            cam_lookat[1] = y
            direction = normalize(cam_lookat - CAM_POSITION)

            origin = CAM_POSITION.copy()
            point, normal, color, shape, min_i = intersect_shapes(
                origin,
                direction,
                cube_centers,
                cube_sizes,
                cube_colors,
                cube_min_bounds,
                cube_max_bounds,
                sphere_centers,
                sphere_radii,
                sphere_colors,
            )
            color_sum += color

        color_sum /= SS_DIM * SS_DIM
        img[-j, i] = np.clip(color_sum, 0, 1)

    return img
