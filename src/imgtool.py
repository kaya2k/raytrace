import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

INPUT_DIR = "img"
OUTPUT_DIR = "img_for_poster"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_and_save(input_path, output_path):
    img = np.array(Image.open(input_path))
    cropped = img[16:-16]
    plt.imsave(output_path, cropped)


if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            crop_and_save(input_path, output_path)
