from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
from src.preparation.utils import np_to_pil, save_image
from threading import Thread
from src.parameters import *
from math import ceil
from colour import Color
import math


BLACK_WHITE = "1"  # (1-bit pixels, black and white, stored with one pixel per byte)
GRAY_SCALE = "L"  # (8-bit pixels, black and white)
COLOR = "RGB"  # (3x8-bit pixels, true color)
COLOR_ALPHA = "RGBA"  # (4x8-bit pixels, true color with transparency mask)
PRINT_STD = "true"

font_ = os.path.join(PREPARATION_FOLDER,"arial.ttf")


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

#
# no gradient RED YELLOW WHITE
#

def blend_np(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_thread, args=(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, ens_prediction, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last(image_np, valid_bit_np, ens_prediction, blended_img)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_thread, args=(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_thread(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    var_norm = ens_prediction[2][index]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (max(ens_prediction[2]) - min(ens_prediction[2]))
    var_mean = 1 - var_norm

    if valid_bit_np[index] == 1:
        color = get_class_color(ens_prediction[0][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_class_color(2)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_last_row_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    var_norm = ens_prediction[2][index]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                    max(ens_prediction[2]) - min(ens_prediction[2]))
    var_mean = 1 - var_norm

    if valid_bit_np[index] == 1:
        color = get_class_color(ens_prediction[0][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_class_color(2)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))


def blend_last(image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    var_norm = ens_prediction[2][index]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    var_mean = 1 - var_norm

    if valid_bit_np[index] == 1:
        color = get_class_color(ens_prediction[0][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_class_color(2)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font=font_, size=20)
    text_color = (0, 0, 0)
    d = ImageDraw.Draw(blended_img)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    var_norm = ens_prediction[2][index]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    var_mean = 1 - var_norm

    if valid_bit_np[index] == 1:
        color = get_class_color(ens_prediction[0][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))
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
        color = get_class_color(2)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, 1), (j * image_np.shape[2], i * image_np.shape[1]))

#
# (prob+(1 - std))/2 gradient RED YELLOW GREEN
#

def blend_np_multiple_gradient(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    gradient = get_color_multiple_gradient()
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
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


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
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


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
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_multiple_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font=font_, size=20)
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
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))

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
        blended_img.paste(blend(base_img, mask, 0.5, 1), (j * image_np.shape[2], i * image_np.shape[1]))

#
#  (1 - std) gradient RED YELLOW WHITE
#

def blend_np_std_gradient(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    print_std_gradient()
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    gradient = get_color_std_gradient()
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_std_gradient_thread, args=(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_std_gradient_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last_std_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_std_gradient_thread, args=(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_std_gradient_thread(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_std_color(gradient[class_idx], var_mean)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_std_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_last_row_std_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_std_color(gradient[class_idx], var_mean)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_std_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_last_std_gradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_std_color(gradient[class_idx], var_mean)
        blended_np[:,:,0].fill(color[0])
        blended_np[:,:,1].fill(color[1])
        blended_np[:,:,2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_std_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_std_gradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font=font_, size=20)
    d = ImageDraw.Draw(blended_img)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_std_color(gradient[class_idx], var_mean)
        text_color = (0,0,0)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))

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
        color = get_std_color(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, 1), (j * image_np.shape[2], i * image_np.shape[1]))

#
#  (1 - std) gradient RED-YELLOW WHITE
#

def blend_np_std_bigradient(image_np, ens_prediction, valid_bit_np, slide_size, print_std, crop_size):
    global PRINT_STD
    PRINT_STD = print_std
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    gradient = get_color_std_bigradient()
    for i in range(0, y_max - 1):
        pool.append(Thread(target=blend_last_column_std_bigradient_thread, args=(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=blend_last_row_std_bigradient_thread, args=(y_max-1 ,j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
        pool[-1].start()
    for p in pool:
        p.join()
    blend_last_std_bigradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=blend_std_bigradient_thread, args=(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def blend_last_column_std_bigradient_thread(i, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color_bigradient(gradient[class_idx], ens_prediction[1][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color_bigradient(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_last_row_std_bigradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)

    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color_bigradient(gradient[class_idx], ens_prediction[1][index])
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color_bigradient(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_last_std_bigradient(image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color_bigradient(gradient[class_idx], ens_prediction[1][index])
        blended_np[:,:,0].fill(color[0])
        blended_np[:,:,1].fill(color[1])
        blended_np[:,:,2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (t_x, t_y))
    else:
        color = get_prob_color_bigradient(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1] - image_np.shape[1]
        blended_img.paste(blend(base_img, mask, 0.5, 1), (t_x, t_y))


def blend_std_bigradient_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img, gradient):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font=font_, size=20)
    d = ImageDraw.Draw(blended_img)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0:
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2])) / (
                max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = 0
    var_mean = 1 - var_norm

    class_idx = ens_prediction[0][index]
    if valid_bit_np[index] == 1:
        color = get_prob_color_bigradient(gradient[class_idx], ens_prediction[1][index])
        text_color = (0,0,0)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))

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
        color = get_prob_color_bigradient(gradient[2], 1)
        blended_np[:, :, 0].fill(color[0])
        blended_np[:, :, 1].fill(color[1])
        blended_np[:, :, 2].fill(color[2])
        mask = np_to_pil(blended_np, COLOR)
        blended_img.paste(blend(base_img, mask, 0.5, 1), (j * image_np.shape[2], i * image_np.shape[1]))


#
# Resize image
#

def save_np_image(image_np, slide_size, crop_size):
    global PRINT_STD
    x_max = ceil(slide_size[0] / crop_size)
    y_max = ceil(slide_size[1] / crop_size)
    bi_x = int(slide_size[0] * image_np.shape[2] / crop_size)
    bi_y = int(slide_size[1] * image_np.shape[1] / crop_size)
    blended_img = Image.new('RGB', (bi_x, bi_y))
    pool = []
    for i in range(0, y_max - 1):
        pool.append(Thread(target=save_last_column_np_thread, args=(i, x_max, image_np, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    pool = []
    for j in range(0, x_max - 1):
        pool.append(Thread(target=save_last_row_np_thread, args=(y_max-1 ,j, x_max, image_np, blended_img)))
        pool[-1].start()
    for p in pool:
        p.join()
    save_last_np(image_np, blended_img)
    pool = []
    for i in range(0, y_max - 1):
        for j in range(0, x_max - 1):
            pool.append(Thread(target=save_np_thread, args=(i, j, x_max, image_np, blended_img)))
            pool[-1].start()
    for p in pool:
        p.join()
    return blended_img


def save_last_column_np_thread(i, x_max, image_np, blended_img):
    index = i * x_max + x_max - 1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    t_x = blended_img.size[0] - image_np.shape[2]
    t_y = i * image_np.shape[1]
    blended_img.paste(base_img, (t_x, t_y))


def save_last_row_np_thread(i, j, x_max, image_np, blended_img):
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    t_x = j * image_np.shape[2]
    t_y = blended_img.size[1] - image_np.shape[1]
    blended_img.paste(base_img, (t_x, t_y))


def save_last_np(image_np, blended_img):
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    t_x = blended_img.size[0] - image_np.shape[2]
    t_y = blended_img.size[1] - image_np.shape[1]
    blended_img.paste(base_img, (t_x, t_y))


def save_np_thread(i, j, x_max, image_np, blended_img):
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))


def get_color_multiple_gradient():
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


def get_color_std_gradient():
    red = Color("OrangeRed")
    dred = Color("DarkRed")
    lgyellow = Color("LightGoldenrodYellow")
    yellow = Color("Yellow")
    white = Color("White")
    gray = Color("#c8c8c8")

    colors_h = list(gray.range_to(white, 101))
    colors_ad = list(lgyellow.range_to(yellow, 101))
    colors_ac = list(red.range_to(dred, 101))
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


def get_color_std_bigradient():
    dred = Color("DarkRed")
    yellow = Color("Yellow")
    lgyellow = Color("LightGoldenrodYellow")
    white = Color("White")
    #gray = Color("#c8c8c8")

    colors_h = list(lgyellow.range_to(white, 69))
    colors_adc = list(lgyellow.range_to(yellow, 42)) + list(yellow.range_to(dred, 27))
    gradient_h = []
    gradient_adc = []
    for c in colors_h:
        gradient_h.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    for c in colors_adc:
        gradient_adc.append([int(c.get_red() * 255), int(c.get_green() * 255), int(c.get_blue() * 255)])
    return [gradient_adc, gradient_adc, gradient_h]


def print_multiple_gradient():
    gradient = get_color_multiple_gradient()
    image = np_to_pil(gradient,COLOR)
    save_image(image,RESOURCE_FOLDER,"gradient")


def print_std_gradient():
    gradient = get_color_std_gradient()
    image = np_to_pil(gradient,COLOR)
    save_image(image,RESOURCE_FOLDER,"std_gradient")


def print_std_bigradient():
    image_np = np.zeros((30,414,3),"uint8")
    gradient = get_color_std_bigradient()
    lst = gradient[2]
    lst.reverse()

    for i in range(0,30):
        for j in range(0,207):
            image_np[i][j][0] = lst[math.floor(j/3)][0]
            image_np[i][j][1] = lst[math.floor(j/3)][1]
            image_np[i][j][2] = lst[math.floor(j/3)][2]
    for i in range(0,30):
        for j in range(207,414):
            idx = math.floor((j-207)/3)
            x = gradient[0][idx][0]
            y = gradient[0][idx][1]
            z = gradient[0][idx][2]
            image_np[i][j][0] = x
            image_np[i][j][1] = y
            image_np[i][j][2] = z
    image = np_to_pil(image_np,COLOR)
    save_image(image,RESOURCE_FOLDER,"std_bigradient")


def get_prob_color(gradient, probability):
    idx = int(probability * 100)
    return gradient[idx]


def get_class_color(class_idx):
    color = [Color("DarkRed"), Color("Yellow"), Color("White")]
    return [int(color[class_idx].get_red() * 255), int(color[class_idx].get_green() * 255), int(color[class_idx].get_blue() * 255)]


def get_prob_color_bigradient(gradient, probability):
    idx = int(probability * 100) - 32
    grad = gradient[idx]
    return grad


def get_std_color(gradient, var_mean):
    idx = int(var_mean * 100)
    return gradient[idx]


def blend(image, mask, factor, unc):
    return Image.blend(image, mask, factor * unc)
