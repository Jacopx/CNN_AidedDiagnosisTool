# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


GRAY_SCALE = "L" # (8-bit pixels, black and white)
COLOR = "RGB" # (3x8-bit pixels, true color)
COLOR_ALPHA = "RGBA" # (4x8-bit pixels, true color with transparency mask)


def make_folder(folder_name):
    try:
        # Create target Directory
        os.makedirs(folder_name)
        print("Directory ", str(folder_name), " Created ")
    except FileExistsError:
        print("Directory ", str(folder_name), " already exists")


def image_to_np_rgb(image):
    return np.asarray(image, dtype="int32")


def np_to_pil(array, mode):
    return Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype="uint8"), mode)


def plot_image(image):
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.show()


def save_image(image, folder, filename):
    make_folder("resources/" + str(folder))
    image.save("resources/" + str(folder) + "/" + str(filename), "PNG")



