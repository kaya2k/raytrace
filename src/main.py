import os
import numpy as np
import time
from matplotlib import pyplot as plt
from core.render import render

cube_centers = []
cube_sizes = []
cube_colors = []
cube_min_bounds = []
cube_max_bounds = []
sphere_centers = []
sphere_radii = []
sphere_colors = []
cylinder_centers = []
cylinder_heights = []
cylinder_radii = []
cylinder_colors = []
round_cube_centers = []
round_cube_rotations = []
sticker_centers = []
sticker_rotations = []
sticker_colors = []


def add_cube(center, size, color):
    size = np.array(size)
    center = np.array(center)
    cube_centers.append(center)
    cube_sizes.append(size)
    cube_colors.append(color)
    cube_min_bounds.append(center - size / 2)
    cube_max_bounds.append(center + size / 2)


def add_sphere(center, radius, color):
    sphere_centers.append(np.array(center))
    sphere_radii.append(radius)
    sphere_colors.append(np.array(color))


def add_cylinder(center, height, radius, color):
    cylinder_centers.append(np.array(center))
    cylinder_heights.append(height)
    cylinder_radii.append(radius)
    cylinder_colors.append(np.array(color))


def add_round_cube(center, rotation):
    round_cube_centers.append(np.array(center))
    round_cube_rotations.append(np.array(rotation))


def add_sticker(center, rotation, color):
    sticker_centers.append(np.array(center))
    sticker_rotations.append(np.array(rotation))
    sticker_colors.append(np.array(color))


def create_scene():
    COLOR_WHITE = (1, 1, 1)
    COLOR_RED = (0.52, 0.14, 0.14)
    COLOR_YELLOW = (0.80, 0.49, 0.15)
    COLOR_SPHERE = COLOR_WHITE
    COLOR_CYLINDER = (0.95, 0.95, 0.90)
    COLOR_STICKER_BLACK = (0.0, 0.0, 0.0)
    COLOR_STICKER_RED = (1.0, 0.0, 0.0)
    COLOR_STICKER_GREEN = (0.32, 0.70, 0.25)
    # Cornell box walls (6 walls) and ceiling panels
    add_cube(center=(0, 0, -100), size=(300, 200, 5), color=COLOR_WHITE)  # back wall
    add_cube(center=(-152.5, 0, 0), size=(5, 200, 205), color=COLOR_YELLOW)  # left wall
    add_cube(center=(152.5, 0, 0), size=(5, 200, 205), color=COLOR_RED)  # right wall
    add_cube(center=(0, -102.5, 0), size=(310, 5, 205), color=COLOR_WHITE)  # floor
    # Ceiling split into four panels, leaving a rectangular opening (the light)
    add_cube(center=(0, 102.5, 62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    add_cube(center=(0, 102.5, -62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    add_cube(center=(92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)
    add_cube(center=(-92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)
    # Spheres (forming a tetrahedron arrangement)
    add_sphere(center=(0, -80.0, 20), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(27.18, -80.0, -9.34), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(-11.8, -80.0, -18.2), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(5.12, -47.34, -2.52), radius=20, color=COLOR_SPHERE)
    # # Cylinder
    add_cylinder(center=(82, -92.5, -10), height=17, radius=24, color=COLOR_CYLINDER)
    # Round cubes
    rotations = [
        [
            [0.7071067811865476, 0.0, -0.7071067811865475],
            [0.27059805007309845, 0.9238795325112867, 0.2705980500730985],
            [0.6532814824381882, -0.3826834323650898, 0.6532814824381883],
        ],
        [
            [0.7071067811865476, 0.0, -0.7071067811865475],
            [-0.2705980500730984, 0.9238795325112867, -0.27059805007309845],
            [0.6532814824381882, 0.38268343236508967, 0.6532814824381883],
        ],
    ]
    centers = [
        (-77.89, -80.84, 18.28),
        (-71.39, -58.67, 24.78),
        (-87.07, -49.48, 9.10),
        (-93.56, -71.66, 2.60),
        (-54.42, -71.66, 7.81),
        (-60.91, -49.48, 1.31),
        (-76.59, -58.67, -14.37),
        (-70.10, -80.84, -7.87),
    ]
    for i in range(8):
        add_round_cube(center=centers[i], rotation=rotations[i // 4])
    # Stickers on round cubes
    rotations = [
        [
            [0.7071067811865476, 0.0, -0.7071067811865475],
            [0.27059805007309845, 0.9238795325112867, 0.2705980500730985],
            [0.6532814824381882, -0.3826834323650898, 0.6532814824381883],
        ],
        [
            [0.7071067811865476, 0.0, -0.7071067811865475],
            [0.6532814824381882, -0.38268343236508967, 0.6532814824381883],
            [-0.2705980500730984, -0.9238795325112867, -0.27059805007309845],
        ],
        [
            [0.7071067811865476, 0.0, -0.7071067811865475],
            [0.6532814824381882, 0.38268343236508967, 0.6532814824381883],
            [0.2705980500730984, -0.9238795325112867, 0.27059805007309845],
        ],
        [
            [0.27059805007309845, 0.9238795325112867, 0.2705980500730985],
            [-0.7071067811865476, 0.0, 0.7071067811865475],
            [0.6532814824381882, -0.3826834323650898, 0.6532814824381883],
        ],
    ]

    centers = [
        (-68.14, -47.58, 28.02),
        (-83.82, -38.40, 12.35),
        (-63.55, -63.26, 32.62),
        (-70.04, -85.43, 26.12),
        (-53.07, -44.89, 9.15),
        (-46.58, -67.06, 15.65),
        (-79.87, -58.67, 33.26),
        (-95.55, -49.48, 17.58),
        (-102.05, -71.66, 11.09),
        (-86.37, -80.84, 26.77),
    ]

    add_sticker(center=centers[0], rotation=rotations[0], color=COLOR_STICKER_BLACK)
    add_sticker(center=centers[1], rotation=rotations[0], color=COLOR_STICKER_BLACK)
    add_sticker(center=centers[2], rotation=rotations[1], color=COLOR_STICKER_RED)
    add_sticker(center=centers[3], rotation=rotations[1], color=COLOR_STICKER_RED)
    add_sticker(center=centers[4], rotation=rotations[2], color=COLOR_STICKER_RED)
    add_sticker(center=centers[5], rotation=rotations[2], color=COLOR_STICKER_RED)
    add_sticker(center=centers[6], rotation=rotations[3], color=COLOR_STICKER_GREEN)
    add_sticker(center=centers[7], rotation=rotations[3], color=COLOR_STICKER_GREEN)
    add_sticker(center=centers[8], rotation=rotations[3], color=COLOR_STICKER_GREEN)
    add_sticker(center=centers[9], rotation=rotations[3], color=COLOR_STICKER_GREEN)


if __name__ == "__main__":
    IMG_PATH = "./img/fig.png"
    if os.path.exists(IMG_PATH):
        print("Rename existing image to avoid overwriting.")
        exit(0)
    create_scene()
    start_time = time.time()
    # Convert scene lists to NumPy arrays and render the image
    img = render(
        np.array(cube_centers, dtype=np.float32),
        np.array(cube_sizes, dtype=np.float32),
        np.array(cube_colors, dtype=np.float32),
        np.array(cube_min_bounds, dtype=np.float32),
        np.array(cube_max_bounds, dtype=np.float32),
        np.array(sphere_centers, dtype=np.float32),
        np.array(sphere_radii, dtype=np.float32),
        np.array(sphere_colors, dtype=np.float32),
        np.array(cylinder_centers, dtype=np.float32),
        np.array(cylinder_heights, dtype=np.float32),
        np.array(cylinder_radii, dtype=np.float32),
        np.array(cylinder_colors, dtype=np.float32),
        np.array(round_cube_centers, dtype=np.float32),
        np.array(round_cube_rotations, dtype=np.float32),
        np.array(sticker_centers, dtype=np.float32),
        np.array(sticker_rotations, dtype=np.float32),
        np.array(sticker_colors, dtype=np.float32),
    )
    end_time = time.time()
    print(f"Rendering took {end_time - start_time:.2f} seconds")
    plt.imsave(IMG_PATH, img)
    print(f"Image saved to {IMG_PATH}")
