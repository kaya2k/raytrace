import numpy as np
from utils import EPSILON, normalize

AMBIENT = 0.20
color_light = np.ones(3)


class Scene:
    def __init__(self):
        self.shapes = []
        self.light_center = np.array((0, 105, 0))
        self.light_x = 58
        self.light_z = 44
        self.light_samples = self.sample_area_light(nx=10, nz=8)

    def sample_area_light(self, nx, nz):
        samples = []
        xs = np.linspace(-self.light_x / 2, self.light_x / 2, nx)
        zs = np.linspace(-self.light_z / 2, self.light_z / 2, nz)
        for i, j in np.ndindex(len(xs), len(zs)):
            x, z = xs[i], zs[j]
            offset = np.array([x, 0, z])
            sample = self.light_center + offset
            samples.append(sample)
        return samples

    def add_shape(self, shape):
        if isinstance(shape, list):
            self.shapes.extend(shape)
        else:
            self.shapes.append(shape)

    def intersect(self, origin, direction):
        closest_distance = float("inf")
        closest_shape = None

        for shape in self.shapes:
            distance = shape.intersect(origin, direction)
            if distance < closest_distance:
                closest_distance = distance
                closest_shape = shape

        if closest_shape is None:
            return

        intersection_point = origin + closest_distance * direction
        normal = closest_shape.get_normal(intersection_point)
        to_origin = normalize(origin - intersection_point)
        color_ray = np.zeros(3)

        for light_pos in self.light_samples:
            to_light = normalize(light_pos - intersection_point)

            # shadow
            shadow_distances = [
                shape.intersect(intersection_point + normal * EPSILON * 10, to_light)
                for shape in self.shapes
            ]
            light_distance = np.linalg.norm(light_pos - intersection_point)
            if shadow_distances and min(shadow_distances) < light_distance:
                continue

            # diffuse and specular contributions
            color_ray += (
                closest_shape.diffuse_c
                * closest_shape.color
                * max(np.dot(normal, to_light), 0)
            )
            color_ray += (
                closest_shape.specular_c
                * color_light
                * max(np.dot(normal, to_origin), 0) ** closest_shape.specular_k
            )

        color_ray /= len(self.light_samples)
        color_ray += AMBIENT * closest_shape.color
        return intersection_point, normal, color_ray
