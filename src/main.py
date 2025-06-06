import numpy as np
import matplotlib.pyplot as plt
from create_scene import create_scene
from tqdm import tqdm
from utils import EPSILON, normalize


MAX_DEPTH = 1
SS_DIM = 2


def main():
    scene = create_scene()

    # img size parameters
    img_ratio = 2000 / 1380
    img_height = 1380 // 10
    img_width = int(img_height * img_ratio)
    img_ratio = img_width / img_height
    img = np.zeros((img_height, img_width, 3), dtype=np.float32)
    print(f"image size: {img_height}x{img_width}")

    # camera parameters
    cam_position = np.array((0, 0, 347.5))
    cam_lookat = np.array((0, 0, 0))
    max_x = 210
    max_y = max_x / img_ratio
    xs = np.linspace(-max_x, max_x, img_width)
    ys = np.linspace(-max_y, max_y, img_height)

    # fill the image
    color = np.zeros(3, dtype=np.float32)
    for i, j in tqdm(np.ndindex(img_width, img_height), total=img_width * img_height):
        color[:] = 0
        for ss in range(SS_DIM * SS_DIM):
            x_size = 2 * max_x / img_width
            y_size = 2 * max_y / img_height
            x = xs[i] + (((ss // SS_DIM) + 0.5) / SS_DIM - 0.5) * x_size
            y = ys[j] + (((ss % SS_DIM) + 0.5) / SS_DIM - 0.5) * y_size
            cam_lookat[:2] = [x, y]

            ray_origin = cam_position
            ray_direction = normalize(cam_lookat - cam_position)

            depth = 0
            reflection = 1.0
            while depth < MAX_DEPTH:
                result = scene.intersect(ray_origin, ray_direction)
                if result is None:
                    break
                intersection_point, normal, color_ray, shape = result
                color += reflection * color_ray

                # prepare next ray
                depth += 1
                reflection *= shape.reflection
                ray_origin = intersection_point + normal * EPSILON * 10
                ray_direction -= 2 * np.dot(ray_direction, normal) * normal
                ray_direction = normalize(ray_direction)

        color /= SS_DIM
        img[-j, i] = np.clip(color, 0, 1)

    plt.imsave(f"./img/{img_height}x{img_width}.png", img)
