# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *            utils.py : utils for images conversion and saving            *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.image as mpimg
import logger as log
import time
from threading import Thread
from src.parameters import *
import math
from colour import Color


BLACK_WHITE = "1"  # (1-bit pixels, black and white, stored with one pixel per byte)
GRAY_SCALE = "L"  # (8-bit pixels, black and white)
COLOR = "RGB"  # (3x8-bit pixels, true color)
COLOR_ALPHA = "RGBA"  # (4x8-bit pixels, true color with transparency mask)
PRINT_STD = "true"

font_ = os.path.join(PREPARATION_FOLDER,"arial.ttf")

def test_folder():
    make_folder(RESOURCE_FOLDER)
    make_folder(DATASET_FOLDER)
    make_folder(MAP_FOLDER)
    make_folder(MODEL_FOLDER)
    make_folder(PREDICTION_FOLDER)
    make_folder(SEGMENTED_FOLDER)
    drop_rates = [0.01, 0.1, 0.5]
    for dr in drop_rates:
        make_folder(path.join("resources", "predictions", str(dr)))


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
    #start_time = time.time()
    image = Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype="uint8"), mode)
    #elapsed_time = time.time() - start_time
    #log.print_debug("Array converted to image || Time Elapsed: " + str(elapsed_time))
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


def plot_var_distribution(file_path, ens_prediction, valid_bit_np):
    log_data_ac = []
    log_data_ad = []
    log_data_h = []
    for i in range(0, len(ens_prediction[0])):
        if valid_bit_np[i]==1:
            if ens_prediction[0][i] == 0:
                if ens_prediction[2][i] != 0:
                    log_data_ac.append(math.log10(ens_prediction[2][i]))

            if ens_prediction[0][i] == 1:
                if ens_prediction[2][i] != 0:
                    log_data_ad.append(math.log10(ens_prediction[2][i]))

            if ens_prediction[0][i] == 2:
                if ens_prediction[2][i] != 0:
                    log_data_h.append(math.log10(ens_prediction[2][i]))

    plt.hist(log_data_ac, bins=25)
    plt.xlabel("Log10(var)")
    plt.title = path.basename(file_path)+"_ac"
    plt.savefig(file_path+"_ac.png")
    plt.clf()
    plt2.hist(log_data_ad, bins=25)
    plt.xlabel("Log10(var)")
    plt2.title = path.basename(file_path)+"_ac"
    plt2.savefig(file_path+"_ad.png")
    plt2.clf()
    plt3.hist(log_data_h, bins=25)
    plt.xlabel("Log10(var)")
    plt3.title = path.basename(file_path)+"_ac"
    plt3.savefig(file_path+"_h.png")
    plt3.clf()


def plot_image(image):
    img = mpimg.imread(image)
    plt.imshow()
    plt.show()


def save_image(image, folder, filename):
    start_time = time.time()
    path = os.path.join(str(folder), str(filename+".png"))
    image.save(path, "PNG")
    elapsed_time = time.time() - start_time
    log.print_debug("Image " + str(path) + " saved || Time Elapsed: " + str(elapsed_time))

