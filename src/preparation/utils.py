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
from threading import Thread
from src.parameters import *
from math import ceil


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


def get_prediction_mask(prediction_matrix, valid_bit_matrix):
    block_size = 224
    prediction_mask = np.zeros((prediction_matrix.shape[0] * block_size, prediction_matrix.shape[1] * block_size, ))
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


def blend_np(image_np, prediction_np, valid_bit_np, slide_size):
    """x_m = x_max*image_np.shape[1]
    y_m = y_max*image_np.shape[2]
    blended_np = np.zeros((y_m, x_m, 3),dtype='uint8')
    if image_np.shape[0] != prediction_np.shape[0] :
        return None
    for k in range(0, image_np.shape[0]):
        for i in range(0,image_np.shape[1]):
            for j in range(0, image_np.shape[2]):
                y = k // x_max * image_np.shape[1] + i
                x = k % x_max * image_np.shape[2] +j
                blended_np[y][x][0] = image_np[k][i][j][0]  # red
                blended_np[y][x][1] = image_np[k][i][j][1]  # green
                blended_np[y][x][2] = image_np[k][i][j][2]  # blue
                if (prediction_np[k][0] > prediction_np[k][1]) and (prediction_np[k][0] > prediction_np[k][2]): # AC
                    if valid_bit_np[k] == 1:
                        blended_np[y][x][0] = 0.8 * blended_np[y][x][0] + 51 # red

                elif (prediction_np[k][1] > prediction_np[k][0]) and (prediction_np[k][1] > prediction_np[k][2]): # AD
                    if valid_bit_np[k] == 1:
                        blended_np[y][x][0] = 0.8 * blended_np[y][x][0] + 51 # red
                        blended_np[y][x][1] = 0.8 * blended_np[y][x][1] + 51 # green
    return blended_np"""
    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)
    bi_x = int(slide_size[0] * image_np.shape[2] / CROP_SIZE)
    bi_y = int(slide_size[1] * image_np.shape[1] / CROP_SIZE)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_thread, args=(i, x_max, image_np, valid_bit_np, prediction_np, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, prediction_np, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last(image_np, valid_bit_np, prediction_np, blended_img)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_thread, args=(i, j, x_max, image_np, valid_bit_np, prediction_np, blended_img)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_thread(i, x_max, image_np, valid_bit_np, prediction_np, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if valid_bit_np[index] == 1:
        if (prediction_np[index][0] > prediction_np[index][1]) and (
                prediction_np[index][0] > prediction_np[index][2]):  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.shape[2] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        elif (prediction_np[index][1] > prediction_np[index][0]) and (
                prediction_np[index][1] > prediction_np[index][2]):  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        else:
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_last_row_thread(i, j, x_max, image_np, valid_bit_np, prediction_np, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if valid_bit_np[index] == 1:
        if (prediction_np[index][0] > prediction_np[index][1]) and (
                prediction_np[index][0] > prediction_np[index][2]):  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        elif (prediction_np[index][1] > prediction_np[index][0]) and (
                prediction_np[index][1] > prediction_np[index][2]):  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        else:
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_last(image_np, valid_bit_np, prediction_np, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if valid_bit_np[index] == 1:
        if (prediction_np[index][0] > prediction_np[index][1]) and (
                prediction_np[index][0] > prediction_np[index][2]):  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask), (j * image_np.shape[2], blended_img.shape[1]-CROP_SIZE))
        elif (prediction_np[index][1] > prediction_np[index][0]) and (
                prediction_np[index][1] > prediction_np[index][2]):  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        else:
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_thread(i, j, x_max, image_np, valid_bit_np, prediction_np, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if valid_bit_np[index] == 1:
        if (prediction_np[index][0] > prediction_np[index][1]) and (
                prediction_np[index][0] > prediction_np[index][2]):  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask), (j * image_np.shape[2], i * image_np.shape[1]))
        elif (prediction_np[index][1] > prediction_np[index][0]) and (
                prediction_np[index][1] > prediction_np[index][2]):  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask), (j * image_np.shape[2], i * image_np.shape[1]))
        else:
            blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))
    else:
        blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))


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