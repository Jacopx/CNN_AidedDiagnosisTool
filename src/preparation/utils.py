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
import matplotlib.image as mpimg
import logger as log
import time
from threading import Thread
from src.parameters import *
from math import ceil
from colour import Color


BLACK_WHITE = "1"  # (1-bit pixels, black and white, stored with one pixel per byte)
GRAY_SCALE = "L"  # (8-bit pixels, black and white)
COLOR = "RGB"  # (3x8-bit pixels, true color)
COLOR_ALPHA = "RGBA"  # (4x8-bit pixels, true color with transparency mask)
PRINT_STD = "true"


def test_folder():
    make_folder(RESOURCE_FOLDER)
    make_folder(DATASET_FOLDER)
    make_folder(MAP_FOLDER)
    make_folder(MODEL_FOLDER)
    make_folder(PREDICTION_FOLDER)
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


def blend_np_gradient(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    gradient = get_color_gradient()
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_gradient_thread, args=(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_gradient_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_gradient_thread, args=(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_gradient_thread(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient, ens_prediction[3][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient, 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))


def blend_last_row_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient, ens_prediction[3][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient, 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))


def blend_last_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                    max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient, ens_prediction[3][index])
        blended_np[:,:,0].fill(color[0])
        blended_np[:,:,1].fill(color[1])
        blended_np[:,:,2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient, 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))


def blend_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font="./arial.ttf", size=20)
    d = ImageDraw.Draw(blended_img)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0 :
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2]) ) / (max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1-var_norm) + ens_prediction[0][index])/ 2

    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient, ens_prediction[3][index])
        text_color = (0,0,0)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))

        k = 10
        location = (j * image_np.shape[2] + 10, i * image_np.shape[1] + k)
        d.text(location, CATEGORIES[prediction_np[index]] + " mean: {:.2}".format(ens_prediction[1][index]),
               font=helvetica, fill=text_color)
        k = k + 30
        if prediction_np[index] != 2:
            location = (j * image_np.shape[2] + 10, i * image_np.shape[1] + k)
            d.text(location, "H mean: {:.2}".format(ens_prediction[3][index]), font=helvetica, fill=text_color)
            k = k + 30
        if PRINT_STD:
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ k)
            d.text(location, "Std: {:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            k = k + 30
        if PRINT_STD:
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ k)
            d.text(location, "Norm std: {:.2%}".format(var_norm) , font=helvetica, fill=text_color)

    else:
        color = get_prob_color(gradient, 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 1), (j * image_np.shape[2], i * image_np.shape[1]))


def blend_np_multiple_gradient(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    gradient = get_color_gradient()
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_multiple_gradient_thread, args=(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_multiple_gradient_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last_multiple_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_multiple_gradient_thread, args=(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_multiple_gradient_thread(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient[class_idx], ens_prediction[1][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))


def blend_last_row_multiple_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient[class_idx], ens_prediction[1][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))


def blend_last_multiple_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                    max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1 - var_norm) + ens_prediction[0][index]) / 2

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient[class_idx], ens_prediction[1][index])
        blended_np[:,:,0].fill(color[0])
        blended_np[:,:,1].fill(color[1])
        blended_np[:,:,2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(blend(base_img, mask, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))


def blend_multiple_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font="arial.ttf", size=20)
    d = ImageDraw.Draw(blended_img)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0 :
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2]) ) / (max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1-var_norm) + ens_prediction[0][index])/ 2

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color(gradient[class_idx], ens_prediction[1][index])
        text_color = (0,0,0)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))

        k = 10
        location = (j * image_np.shape[2] + 10, i * image_np.shape[1] + k)
        d.text(location, CATEGORIES[ens_prediction[0][index]] + " mean: {:.2}".format(ens_prediction[1][index]),
               font=helvetica, fill=text_color)
        k = k + 30
        if ens_prediction[0][index] != 2:
            location = (j * image_np.shape[2] + 10, i * image_np.shape[1] + k)
            d.text(location, "H mean: {:.2}".format(ens_prediction[3][index]), font=helvetica, fill=text_color)
            k = k + 30
        if PRINT_STD:
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ k)
            d.text(location, "Std: {:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            k = k + 30
        if PRINT_STD:
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ k)
            d.text(location, "Norm std: {:.2%}".format(var_norm) , font=helvetica, fill=text_color)

    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 1), (j * image_np.shape[2], i * image_np.shape[1]))



"""def get_color_gradient():
    red = Color("Red")
    orange = Color("Orange")
    yellow = Color("Yellow")
    violet = Color("Violet")
    colors = list()
    colors = colors + list(red.range_to(orange, 30))
    colors = colors + list(orange.range_to(yellow, 30))
    colors = colors + list(yellow.range_to(violet, 41))
    gradient = []
    for c in colors:
        gradient.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    return gradient
"""

def get_color_gradient():
    red = Color("Red")
    orange = Color("Orange")
    yellow = Color("Yellow")
    green = Color("Green")
    colors_h = list(yellow.range_to(green, 101))
    colors_ad = list(yellow.range_to(orange, 101))
    colors_ac = list(orange.range_to(red, 101))
    gradient_h = []
    gradient_ad = []
    gradient_ac = []
    for c in colors_h:
        gradient_h.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    for c in colors_ad:
        gradient_ad.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    for c in colors_ac:
        gradient_ac.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    return [gradient_ac, gradient_ad, gradient_h]


def print_gradient():
    gradient = get_color_gradient()
    image = np_to_pil(gradient,COLOR)
    save_image(image,RESOURCE_FOLDER,"gradient")


def get_prob_color(gradient, probability):
    idx = int(probability * 100)
    return gradient[idx]


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


def blend(image, mask, unc):
    return Image.blend(image, mask, 0.5 * unc)
