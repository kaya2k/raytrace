import numpy as np
from shape import Shape
from utils import EPSILON, dot, normalize


AMBIENT = 0.10
NX = 10
NZ = int(NX * 44 / 58)


class Scene:
    def __init__(self):
        self.shapes = []
        self.light_center = np.array((0, 105, 0))
        self.light_x = 58
        self.light_z = 44
        self.light_samples = self.sample_area_light(nx=NX, nz=NZ)
        print(f"light samples: {NX}x{NZ}")

    def sample_area_light(self, nx, nz):
        samples = []
        for i, j in np.ndindex(nx, nz):
            x = -self.light_x / 2 + self.light_x / (nx + 1) * (i + 1)
            z = -self.light_z / 2 + self.light_z / (nz + 1) * (j + 1)
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
        distance, shape = self.find_closest_intersection(origin, direction)
        if shape is None:
            return
        point = origin + distance * direction
        normal = shape.get_normal(point)
        color = self.compute_color(point, normal, normalize(origin - point), shape)
        return point, normal, color, shape

    def find_closest_intersection(
        self, origin, direction
    ) -> tuple[float, Shape | None]:
        closest_distance = float("inf")
        closest_shape = None

        for shape in self.shapes:
            distance = shape.intersect(origin, direction)
            if distance < closest_distance:
                closest_distance = distance
                closest_shape = shape

        return closest_distance, closest_shape

    def compute_color(self, point, normal, to_origin, shape):
        color = np.zeros(3)
        for light_pos in self.light_samples:
            to_light = normalize(light_pos - point)
            light_distance = np.linalg.norm(light_pos - point)
            in_shadow = False

            for s in self.shapes:
                distance = s.intersect(point + to_light * EPSILON * 10, to_light)
                if distance < light_distance:
                    in_shadow = True
                    break

            if not in_shadow:
                color += (
                    shape.DIFF_C * shape.color * dot(normal, to_light)
                    + shape.SPEC_C * np.ones(3) * dot(normal, to_origin) ** shape.SPEC_K
                )

        color /= len(self.light_samples)
        color += AMBIENT * shape.color
        return color
