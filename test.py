import numpy as np

LIGHT_SAMPLES_PER_CELL = 4
LIGHT_Y = 105
LIGHT_LEN_X = 58
LIGHT_LEN_Z = 44
LIGHT_NX = 3
LIGHT_NZ = int(LIGHT_NX * LIGHT_LEN_Z / LIGHT_LEN_X)
CELL_SIZE_X = LIGHT_LEN_X / LIGHT_NX
CELL_SIZE_Z = LIGHT_LEN_Z / LIGHT_NZ


print(f"LIGHT_NX: {LIGHT_NX}, LIGHT_NZ: {LIGHT_NZ}")
print(f"CELL_SIZE_X: {CELL_SIZE_X}, CELL_SIZE_Z: {CELL_SIZE_Z}")


for idx in range(LIGHT_NX * LIGHT_NZ):
    x_min = (idx // LIGHT_NZ) * CELL_SIZE_X
    z_min = (idx % LIGHT_NZ) * CELL_SIZE_Z

    for _ in range(LIGHT_SAMPLES_PER_CELL):
        x = x_min + np.random.rand() * CELL_SIZE_X
        z = z_min + np.random.rand() * CELL_SIZE_Z
        light_sample = np.array([x, LIGHT_Y, z])
        print(f"Light sample: {x}, {LIGHT_Y}, {z}")
