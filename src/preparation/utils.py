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
from src.parameters import *


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


def list_np_to_pil(image_np, mode):
    start_time = time.time()
    print(str(image_np.shape[0]*image_np.shape[2])+" "+str(image_np.shape[1]*image_np.shape[3]))
    array = np.zeros((image_np.shape[0]*image_np.shape[2], image_np.shape[1]*image_np.shape[3], 3))
    for i in range(0, image_np.shape[0]):
        for j in range(0, image_np.shape[1]):
            for k in range(0, image_np.shape[2]):
                for t in range(0, image_np.shape[3]):
                    for z in range(0, image_np.shape[4]):
                        array[i*image_np.shape[2]+k][j*image_np.shape[3]+t][z]=image_np[i][j][k][t][z]
    image = Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype="uint8"), mode)
    elapsed_time = time.time() - start_time
    log.print_debug("Array converted to image || Time Elapsed: " + str(elapsed_time))
    return image


def list_np_to_pil_test(image_np, mode, x_max, y_max):
    start_time = time.time()

    array = np.zeros((y_max*image_np.shape[1], x_max*image_np.shape[2], 3))
    for i in range(0, image_np.shape[0]):
         for j in range(0, image_np.shape[1]):
            for k in range(0, image_np.shape[2]):
                x = i // x_max * image_np.shape[1] + j
                y = i % x_max * image_np.shape[2] + k
                for t in range(0, image_np.shape[3]):
                    array[x][y][t] = image_np[i][j][k][t]

    image = Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype="uint8"), mode)
    elapsed_time = time.time() - start_time
    log.print_debug("Array converted to image || Time Elapsed: " + str(elapsed_time))
    return image


def plot_histogram_lum(image):
    plt.hist(image, bins=255)
    plt.title("histogram")
    plt.show()


def get_prediction_mask(prediction_matrix, valid_bit_matrix):
    block_size = 224
    prediction_mask = np.zeros((prediction_matrix.shape[0] * block_size, prediction_matrix.shape[1] * block_size, 4))
    print(prediction_matrix.shape)
    for i in range(0, prediction_mask.shape[0]):
        for j in range(0, prediction_mask.shape[1]):
            y = i // block_size
            x = j // block_size
            if(prediction_matrix[y][x][0] > prediction_matrix[y][x][1]) and (prediction_matrix[y][x][0] > prediction_matrix[y][x][2]): #AC
                prediction_mask[i][j][0] = 255 # red
                prediction_mask[i][j][1] = 0 # green
                prediction_mask[i][j][2] = 0   # blue
                if valid_bit_matrix[y][x] == 1:
                    prediction_mask[i][j][3] = 128 # alpha
                else:
                    prediction_mask[i][j][3] = 0 # alpha
            elif(prediction_matrix[y][x][1] > prediction_matrix[y][x][0]) and (prediction_matrix[y][x][1] > prediction_matrix[y][x][2]): #AD
                prediction_mask[i][j][0] = 255  # red
                prediction_mask[i][j][1] = 255  # green
                prediction_mask[i][j][2] = 0  # blue
                if valid_bit_matrix[y][x] == 1:
                    prediction_mask[i][j][3] = 128 # alpha
                else:
                    prediction_mask[i][j][0] = 255  # red
                    prediction_mask[i][j][1] = 255  # green
                    prediction_mask[i][j][2] = 255  # blue
                    prediction_mask[i][j][3] = 0 # alpha
            else:
                prediction_mask[i][j][0] = 255  # red
                prediction_mask[i][j][1] = 255  # green
                prediction_mask[i][j][2] = 255  # blue
                prediction_mask[i][j][3] = 0  # alpha
    return prediction_mask


def plot_image(image):
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.show()


def save_image(image, folder, filename):
    start_time = time.time()
    path = os.path.join(str(folder), str(filename+".png"))
    image.save(path, "PNG")
    elapsed_time = time.time() - start_time
    log.print_debug("Image " + str(path) + " saved || Time Elapsed: " + str(elapsed_time))


def blend(image, mask):
    return Image.blend(image, mask, 0.2)