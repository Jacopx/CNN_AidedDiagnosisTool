# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *            utils.py : utils for images conversion and saving            *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logger as log
import time


BLACK_WHITE = "1"  # (1-bit pixels, black and white, stored with one pixel per byte)
GRAY_SCALE = "L"  # (8-bit pixels, black and white)
COLOR = "RGB"  # (3x8-bit pixels, true color)
COLOR_ALPHA = "RGBA"  # (4x8-bit pixels, true color with transparency mask)


def make_folder(folder_name):
    try:
        # Create target Directory
        os.makedirs(folder_name)
        log.print_info(" Directory " + str(folder_name) + " created ")
    except FileExistsError:
        log.print_info(" Directory " + str(folder_name) + " already exists")


def image_to_np_rgb(image):
    start_time = time.time()
    np_image = np.asarray(image, dtype="int32")
    elapsed_time = time.time() - start_time
    log.print_debug("Image converted to array || Time Elapsed: " + str(elapsed_time))
    return np_image


def np_to_pil(array, mode):
    start_time = time.time()
    image = Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype="uint8"), mode)
    elapsed_time = time.time() - start_time
    log.print_debug("Array converted to image || Time Elapsed: " + str(elapsed_time))
    return image


def plot_histogram_lum(image):
    plt.hist(image, bins=255)
    plt.title("histogram")
    plt.show()


def plot_image(image):
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.show()


def save_image(image, folder, filename):
    start_time = time.time()
    path = os.path.join(str(folder), str(filename))
    image.save(path, "PNG")
    elapsed_time = time.time() - start_time
    log.print_debug("Image " + str(path) + " saved || Time Elapsed: " + str(elapsed_time))
