import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == "__main__":
    IMG_PATH = "img/MIRROR.png"
    NEW_PATH = "img/NEW.png"
    img = np.array(Image.open(IMG_PATH))
    img[:16] = 0.3
    img[-16:] = 0.3
    plt.imsave(NEW_PATH, img)
