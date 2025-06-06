import numpy as np

EPSILON = 1e-6


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dot(a, b):
    return max(np.dot(a, b), 0.0)
