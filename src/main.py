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


def create_scene():
    COLOR_WHITE = (1, 1, 1)
    COLOR_RED = (0.52, 0.14, 0.14)
    COLOR_YELLOW = (0.80, 0.49, 0.15)
    COLOR_SPHERE = (0.93, 0.94, 0.98)

    # add cornell box
    add_cube(center=(0, 0, -100), size=(300, 200, 5), color=COLOR_WHITE)
    add_cube(center=(-152.5, 0, 0), size=(5, 200, 205), color=COLOR_YELLOW)
    add_cube(center=(152.5, 0, 0), size=(5, 200, 205), color=COLOR_RED)
    add_cube(center=(0, -102.5, 0), size=(310, 5, 205), color=COLOR_WHITE)
    add_cube(center=(0, 102.5, 62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    add_cube(center=(0, 102.5, -62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    add_cube(center=(92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)
    add_cube(center=(-92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)

    # add tetrahedron
    add_sphere(center=(0, -80, 20), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(27.18, -80.0, -9.34), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(-11.82, -80.0, -18.21), radius=20, color=COLOR_SPHERE)
    add_sphere(center=(5.12, -47.34, -2.52), radius=20, color=COLOR_SPHERE)


if __name__ == "__main__":
    IMG_PATH = "./img/fig.png"
    if os.path.exists(IMG_PATH):
        print("Rename existing image to avoid overwriting.")
        exit(0)

    create_scene()
    start_time = time.time()
    img = render(
        np.array(cube_centers, dtype=np.float32),
        np.array(cube_sizes, dtype=np.float32),
        np.array(cube_colors, dtype=np.float32),
        np.array(cube_min_bounds, dtype=np.float32),
        np.array(cube_max_bounds, dtype=np.float32),
        np.array(sphere_centers, dtype=np.float32),
        np.array(sphere_radii, dtype=np.float32),
        np.array(sphere_colors, dtype=np.float32),
    )
    end_time = time.time()
    print(f"Rendering took {end_time - start_time:.2f} seconds")
    plt.imsave(IMG_PATH, img)
    print(f"Image saved to {IMG_PATH}")
