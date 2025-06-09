import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == "__main__":
    NEW_PATH = "img/PT8192_D10_LP21_ENV03.png"

    IMG_PATH = "img/000.png"
    LP_PATH = "data/PT8192_D10_LP20_ENV00.npy"
    EP_PATH = "data/PT0256_D10_LP00_ENV10.npy"
    ref_img = np.array(Image.open(IMG_PATH))
    lp_img = np.load(LP_PATH)
    ep_img = np.load(EP_PATH)
    while True:
        w_lp = float(input("Enter LP Weight(0 to quit): "))
        w_ep = float(input("Enter EP Weight(0 to quit): "))
        new_img = np.clip(ep_img * w_ep + lp_img * w_lp, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14 * 1.5, 5 * 1.5), facecolor="black")
        axes = axes.ravel()

        images = [ref_img, new_img]
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.imsave(NEW_PATH, new_img)
