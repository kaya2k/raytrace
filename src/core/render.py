import math
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from .scene import trace_ray_device

# Image and camera parameters (same as original)
IMG_RATIO  = 2000 / 1380
IMG_WIDTH  = 2000
IMG_HEIGHT = int(IMG_WIDTH / IMG_RATIO)
IMG_RATIO  = IMG_WIDTH / IMG_HEIGHT
CAM_POSITION = np.array((0.0, 0.0, 347.5), dtype=np.float32)
CAM_MAX_X   = 210.0
CAM_MAX_Y   = CAM_MAX_X / IMG_RATIO
CELL_SIZE_X = CAM_MAX_X * 2.0 / IMG_WIDTH
CELL_SIZE_Y = CAM_MAX_Y * 2.0 / IMG_HEIGHT
SS_DIM = 2  # supersampling dimension (2x2)

@cuda.jit
def render_kernel(img,
                 cube_centers, cube_sizes, cube_colors,
                 cube_min_bounds, cube_max_bounds,
                 sphere_centers, sphere_radii, sphere_colors,
                 states, n_cubes, n_spheres):
    # 2D thread indices for pixel coordinates
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # image column (x index)
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  # image row (y index)
    if i < img.shape[1] and j < img.shape[0]:
        # Determine output image row index (flip vertically to match original orientation)
        # (Note: -0 in Python is 0, so j==0 -> out_y = 0, otherwise out_y = height - j)
        out_y = 0 if j == 0 else img.shape[0] - j
        # Compute the base coordinates (top-left corner) of the pixel in camera plane
        # (Camera looks towards Z=0 plane, with FOV defined by CAM_MAX_X, CAM_MAX_Y)
        min_x = i * CELL_SIZE_X - CAM_MAX_X
        min_y = j * CELL_SIZE_Y - CAM_MAX_Y
        # Initialize color accumulator
        float_r = 0.0
        float_g = 0.0
        float_b = 0.0
        # 2x2 supersampling loop
        for k in range(SS_DIM):
            for l in range(SS_DIM):
                # Sample position within the pixel (center of each subcell)
                x = min_x + (k + 0.5) / SS_DIM * CELL_SIZE_X
                y = min_y + (l + 0.5) / SS_DIM * CELL_SIZE_Y
                # Camera is looking toward (x, y, 0) from CAM_POSITION
                dir_x = x - CAM_POSITION[0]
                dir_y = y - CAM_POSITION[1]
                dir_z = 0.0 - CAM_POSITION[2]
                # Normalize the direction vector
                length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
                if length > 0.0:
                    inv_len = 1.0 / length
                    dir_x *= inv_len
                    dir_y *= inv_len
                    dir_z *= inv_len
                # Trace the ray through the scene and get the color
                r, g, b = trace_ray_device(CAM_POSITION[0], CAM_POSITION[1], CAM_POSITION[2],
                                           dir_x, dir_y, dir_z,
                                           cube_centers, cube_sizes, cube_colors,
                                           cube_min_bounds, cube_max_bounds,
                                           sphere_centers, sphere_radii, sphere_colors,
                                           states, j * img.shape[1] + i,  # unique RNG index
                                           n_cubes, n_spheres)
                float_r += r
                float_g += g
                float_b += b
        # Average the sub-pixel samples
        inv_ss = 1.0 / float(SS_DIM * SS_DIM)
        float_r *= inv_ss
        float_g *= inv_ss
        float_b *= inv_ss
        # Clamp color components to [0, 1]
        if float_r < 0.0: float_r = 0.0
        if float_r > 1.0: float_r = 1.0
        if float_g < 0.0: float_g = 0.0
        if float_g > 1.0: float_g = 1.0
        if float_b < 0.0: float_b = 0.0
        if float_b > 1.0: float_b = 1.0
        # Write the color to the output image (as float32)
        img[out_y, i, 0] = float_r
        img[out_y, i, 1] = float_g
        img[out_y, i, 2] = float_b

def render(cube_centers, cube_sizes, cube_colors,
           cube_min_bounds, cube_max_bounds,
           sphere_centers, sphere_radii, sphere_colors):
    # Allocate device memory and copy input data
    d_cube_centers   = cuda.to_device(cube_centers)
    d_cube_sizes     = cuda.to_device(cube_sizes)
    d_cube_colors    = cuda.to_device(cube_colors)
    d_cube_min_bounds= cuda.to_device(cube_min_bounds)
    d_cube_max_bounds= cuda.to_device(cube_max_bounds)
    d_sphere_centers = cuda.to_device(sphere_centers)
    d_sphere_radii   = cuda.to_device(sphere_radii)
    d_sphere_colors  = cuda.to_device(sphere_colors)
    # Prepare output image array on device
    d_img = cuda.device_array((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    # Initialize random states for each thread (for area light sampling)
    n_pixels = IMG_WIDTH * IMG_HEIGHT
    # Use a fixed seed for reproducibility (same sequence each run)
    states = create_xoroshiro128p_states(n_pixels, seed=42)
    # Configure CUDA grid and block dimensions
    threads_per_block = (16, 16)
    blocks_x = math.ceil(IMG_WIDTH / threads_per_block[0])
    blocks_y = math.ceil(IMG_HEIGHT / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)
    # Launch the kernel
    render_kernel[blocks_per_grid, threads_per_block](
        d_img,
        d_cube_centers, d_cube_sizes, d_cube_colors,
        d_cube_min_bounds, d_cube_max_bounds,
        d_sphere_centers, d_sphere_radii, d_sphere_colors,
        states, cube_centers.shape[0], sphere_centers.shape[0]
    )
    # Wait for GPU to finish
    cuda.synchronize()
    # Copy the rendered image back to host as a NumPy array
    img_host = d_img.copy_to_host()
    return img_host
