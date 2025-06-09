import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

IMG_PATH = "img/000.png"
LP_PATH = "data/PT8192_D10_LP20_ENV00.npy"
EP_PATH = "data/PT0256_D10_LP00_ENV10.npy"
ref = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float32) / 255.0
lp = np.load(LP_PATH)
env = np.load(EP_PATH)

min_mse = float("inf")
best_weights = (0, 0)
best_img = None


fig, axes = plt.subplots(3, 3, figsize=(20 * 0.8, 14 * 0.8), facecolor="black")
axes = axes.ravel()

for i in range(9):
    x = i / 9
    w_light = 1.05 + x * 0.2
    w_env = 0.3
    new_img = np.clip(env * w_env + lp * w_light, 0, 1)

    ax = axes[i]
    ax.imshow(new_img)
    ax.axis("off")

plt.tight_layout()
plt.show()
