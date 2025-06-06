from shape import Cube, Sphere
from scene import Scene


def create_scene():
    scene = Scene()

    # add cornell box
    COLOR_WHITE = (1, 1, 1)
    COLOR_RED = (130 / 255, 0, 0)
    COLOR_YELLOW = (217 / 255, 170 / 255, 0)
    wall_B = Cube(center=(0, 0, -100), size=(300, 200, 5), color=COLOR_WHITE)
    wall_L = Cube(center=(-152.5, 0, 0), size=(5, 200, 205), color=COLOR_YELLOW)
    wall_R = Cube(center=(152.5, 0, 0), size=(5, 200, 205), color=COLOR_RED)
    wall_D = Cube(center=(0, -102.5, 0), size=(310, 5, 205), color=COLOR_WHITE)
    wall_U1 = Cube(center=(0, 102.5, 62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    wall_U2 = Cube(center=(0, 102.5, -62.25), size=(310, 5, 80.5), color=COLOR_WHITE)
    wall_U3 = Cube(center=(92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)
    wall_U4 = Cube(center=(-92, 102.5, 0), size=(126, 5, 44), color=COLOR_WHITE)
    scene.add_shape([wall_B, wall_L, wall_R, wall_D])
    scene.add_shape([wall_U1, wall_U2, wall_U3, wall_U4])

    # add tetrahedron
    COLOR_SPHERE = (235 / 255, 235 / 255, 235 / 255)
    sphere_0 = Sphere(center=(0, -80, 20), radius=20, color=COLOR_SPHERE)
    sphere_1 = Sphere(center=(27.18, -80.0, -9.34), radius=20, color=COLOR_SPHERE)
    sphere_2 = Sphere(center=(-11.82, -80.0, -18.21), radius=20, color=COLOR_SPHERE)
    sphere_3 = Sphere(center=(5.12, -47.34, -2.52), radius=20, color=COLOR_SPHERE)
    scene.add_shape([sphere_0, sphere_1, sphere_2, sphere_3])

    return scene
