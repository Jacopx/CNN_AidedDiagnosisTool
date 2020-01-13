import os
import glob
import random
from src.preparation import filter, slide, utils
import logger as log
from PIL import Image
import numpy as np
import pickle
from src.cnn import testCNN as cnn
from threading import Thread
from math import ceil
from src.parameters import *

def custom_crop(slide_, y, batch_to_predict, valid_bit_matrix, x_max):
    batch_to_predict[y] = []
    valid_bit_matrix[y] = []
    for x in range(0, x_max-1):
        image_crop = slide.read_slide_crop(slide_, x * CROP_SIZE, y * CROP_SIZE).convert('RGB')
        resized_crop = slide.resize_image_a(image_crop, 224, 224)
        resized_crop_np = np.asarray(resized_crop, np.float32)
        if filter.check_valid(resized_crop_np):
            valid_bit_matrix[y].append(1)
        else:
            valid_bit_matrix[y].append(0)
        batch_to_predict[y].append(resized_crop_np)
    image_crop = slide.read_slide_crop(slide_, slide.get_slide_size(slide_)[0] - CROP_SIZE, y * CROP_SIZE).convert('RGB')
    resized_crop = slide.resize_image_a(image_crop, 224, 224)
    resized_crop_np = np.asarray(resized_crop, np.float32)
    if filter.check_valid(resized_crop_np):
        valid_bit_matrix[y].append(1)
    else:
        valid_bit_matrix[y].append(0)
    batch_to_predict[y].append(resized_crop_np)


def custom_crop_last(slide_, y, batch_to_predict, valid_bit_matrix, x_max):
    batch_to_predict[y] = []
    valid_bit_matrix[y] = []
    for x in range(0, x_max-1):
        image_crop = slide.read_slide_crop(slide_, x * CROP_SIZE, slide.get_slide_size(slide_)[1] * CROP_SIZE).convert('RGB')
        resized_crop = slide.resize_image_a(image_crop, 224, 224)
        resized_crop_np = np.asarray(resized_crop, np.float32)
        if filter.check_valid(resized_crop_np):
            valid_bit_matrix[y].append(1)
        else:
            valid_bit_matrix[y].append(0)
        batch_to_predict[y].append(resized_crop_np)
    image_crop = slide.read_slide_crop(slide_, slide.get_slide_size(slide_)[0] - CROP_SIZE, slide.get_slide_size(slide_)[1] * CROP_SIZE).convert('RGB')
    resized_crop = slide.resize_image_a(image_crop, 224, 224)
    resized_crop_np = np.asarray(resized_crop, np.float32)
    if filter.check_valid(resized_crop_np):
        valid_bit_matrix[y].append(1)
    else:
        valid_bit_matrix[y].append(0)
    batch_to_predict[y].append(resized_crop_np)


def make_prediction_thread(batch_to_predict_np,prediction_list):
    prediction_list = cnn.predict_from_model(batch_to_predict_np)


def get_prediction_matrix_multithread(slidename):
    slide_ = slide.open_slide(os.path.join(MAP_FOLDER, slidename))
    slide_size = slide.get_slide_size(slide_)
    log.print_info("Slide size : " + str(slide_size))

    pool = []

    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)

    log.print_info("Matrix size : " + str(y_max) + " " + str(x_max))
    batch_to_predict = [None] * y_max
    valid_bit_matrix = [None] * y_max
    for y in range(0, y_max-1):
        pool.append(Thread(target=custom_crop, args=(slide_, y, batch_to_predict, valid_bit_matrix, x_max)))
        pool[-1].start()
    t = Thread(target=custom_crop_last, args=(slide_, y_max - 1, batch_to_predict, valid_bit_matrix, x_max))
    t.start()
    for p in pool:
        p.join()
    t.join()

    batch_to_predict_np = np.asarray(batch_to_predict)
    print(batch_to_predict_np.shape)
    prediction_list = [[]] * batch_to_predict_np.shape[0]
    for i in range(0, batch_to_predict_np.shape[0]):
       cnn.predict_from_model_multithread(batch_to_predict_np[i][:][:][:], prediction_list, i)
    prediction_matrix = np.asarray(prediction_list)
    print(prediction_matrix.shape)
    return utils.list_np_to_pil(batch_to_predict_np,utils.COLOR), prediction_matrix, valid_bit_matrix


def custom_crop_test(slide_, y, batch_to_predict, valid_bit_matrix, x_max):
    valid_bit_matrix[y] = []
    for x in range(0, x_max-1):
        image_crop = slide.read_slide_crop(slide_, x * CROP_SIZE, y * CROP_SIZE).convert('RGB')
        resized_crop = slide.resize_image_a(image_crop, 224, 224)
        resized_crop_np = np.asarray(resized_crop, np.float32)
        if filter.check_valid(resized_crop_np):
            valid_bit_matrix[y].append(1)
        else:
            valid_bit_matrix[y].append(0)
        batch_to_predict[y*x_max+x] = resized_crop_np
    image_crop = slide.read_slide_crop(slide_, slide.get_slide_size(slide_)[0] - CROP_SIZE, y * CROP_SIZE).convert('RGB')
    resized_crop = slide.resize_image_a(image_crop, 224, 224)
    resized_crop_np = np.asarray(resized_crop, np.float32)
    if filter.check_valid(resized_crop_np):
        valid_bit_matrix[y].append(1)
    else:
        valid_bit_matrix[y].append(0)
    batch_to_predict[y*x_max+x_max-1] = resized_crop_np


def custom_crop_last_test(slide_, y, batch_to_predict, valid_bit_matrix, x_max):
    batch_to_predict[y] = []
    valid_bit_matrix[y] = []
    for x in range(0, x_max-1):
        image_crop = slide.read_slide_crop(slide_, x * CROP_SIZE, slide.get_slide_size(slide_)[1] * CROP_SIZE).convert('RGB')
        resized_crop = slide.resize_image_a(image_crop, 224, 224)
        resized_crop_np = np.asarray(resized_crop, np.float32)
        if filter.check_valid(resized_crop_np):
            valid_bit_matrix[y].append(1)
        else:
            valid_bit_matrix[y].append(0)
        batch_to_predict[y*x_max+x] = resized_crop_np
    image_crop = slide.read_slide_crop(slide_, slide.get_slide_size(slide_)[0] - CROP_SIZE, slide.get_slide_size(slide_)[1] * CROP_SIZE).convert('RGB')
    resized_crop = slide.resize_image_a(image_crop, 224, 224)
    resized_crop_np = np.asarray(resized_crop, np.float32)
    if filter.check_valid(resized_crop_np):
        valid_bit_matrix[y].append(1)
    else:
        valid_bit_matrix[y].append(0)
    batch_to_predict[y*x_max+x_max-1] = resized_crop_np


def get_prediction_matrix_multithread_test(slidename):
    slide_ = slide.open_slide(os.path.join(MAP_FOLDER, slidename))
    slide_size = slide.get_slide_size(slide_)
    log.print_info("Slide size : " + str(slide_size))

    pool = []

    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)

    log.print_info("Matrix size : " + str(y_max) + " " + str(x_max))
    batch_to_predict = [None] * x_max * y_max
    valid_bit_matrix = [None] * y_max
    for y in range(0, y_max-1):
        pool.append(Thread(target=custom_crop_test, args=(slide_, y, batch_to_predict, valid_bit_matrix, x_max)))
        pool[-1].start()
    t = Thread(target=custom_crop_last_test, args=(slide_, y_max - 1, batch_to_predict, valid_bit_matrix, x_max))
    t.start()
    for p in pool:
        p.join()
    t.join()

    batch_to_predict_np = np.asarray(batch_to_predict)
    print(batch_to_predict_np.shape)
    prediction_list = cnn.predict_from_model_multithread_temp(batch_to_predict_np)
    prediction_matrix = np.asarray(prediction_list)
    print(prediction_matrix.shape)
    return utils.list_np_to_pil_test(batch_to_predict_np,utils.COLOR, x_max, y_max).convert('RGBA'), np.reshape(prediction_matrix,(y_max,x_max,3)), valid_bit_matrix


"""def get_prediction_matrix_multithread_test2(slidename):
    slide_ = slide.open_slide(os.path.join(MAP_FOLDER, slidename))
    slide_size = slide.get_slide_size(slide_)
    log.print_info("Slide size : " + str(slide_size))
    pool = []
    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)
    log.print_info("Matrix size : " + str(y_max) + " " + str(x_max))
    batch_to_predict = [None] * x_max * y_max
    valid_bit_list = [None] * x_max *  y_max
    for y in range(0, y_max-1):
        pool.append(Thread(target=custom_crop_test2, args=(slide_, y, batch_to_predict, valid_bit_list, x_max)))
        pool[-1].start()
    t = Thread(target=custom_crop_last_test2, args=(slide_, y_max - 1, batch_to_predict, valid_bit_list, x_max))
    t.start()
    for p in pool:
        p.join()
    t.join()
    valid_bit_np=np.asarray(valid_bit_list)
    batch_to_predict_np = np.asarray(batch_to_predict)
    print(batch_to_predict_np.shape)
    print(valid_bit_np.shape)
    print("-----------------------------")
    prediction_list = cnn.predict_from_model_multithread_temp(batch_to_predict_np)
    prediction_np = np.asarray(prediction_list)
    print(batch_to_predict_np.shape)
    print(valid_bit_np.shape)
    print(prediction_np.shape)
    print("-----------------------------")
    return utils.blend_np(batch_to_predict_np, prediction_np, valid_bit_np, slide_size)"""




"""def blend_np(image_np, prediction_np, valid_bit_np, slide_size):
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
    return blended_img"""


"""def blend_np_var(image_np, ens_prediction, valid_bit_np, slide_size):
    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)
    bi_x = int(slide_size[0] * image_np.shape[2] / CROP_SIZE)
    bi_y = int(slide_size[1] * image_np.shape[1] / CROP_SIZE)
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
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0 :  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        elif prediction_np[index] == 1 :  # AD
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


def blend_last_row_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0:  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        elif prediction_np[index] == 1:  # AD
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


def blend_last(image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0 :  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask), (t_x, t_y))
        elif prediction_np[index] == 1:  # AD
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


def blend_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font="arial.ttf", size=30)
    d = ImageDraw.Draw(blended_img)
    text_color = (0, 0, 0)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0:  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask), (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "{:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 50)
            d.text(location, "{:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
        elif prediction_np[index] == 1:  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask), (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "{:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 50)
            d.text(location, "{:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
        else:
            blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "{:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 50)
            d.text(location, "{:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
    else:
        blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))
"""

"""def blend_np_var(image_np, ens_prediction, valid_bit_np, slide_size):
    x_max = ceil(slide_size[0] / CROP_SIZE)
    y_max = ceil(slide_size[1] / CROP_SIZE)
    bi_x = int(slide_size[0] * image_np.shape[2] / CROP_SIZE)
    bi_y = int(slide_size[1] * image_np.shape[1] / CROP_SIZE)
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
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0 :  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        elif prediction_np[index] == 1 :  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        else:
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = i * image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = i * image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_last_row_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0:  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        elif prediction_np[index] == 1:  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        else:
            t_x = j * image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = j * image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_last(image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    index = -1
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0 :  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        elif prediction_np[index] == 1:  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(blend(base_img, mask, 1), (t_x, t_y))
        else:
            t_x = blended_img.size[0] - image_np.shape[2]
            t_y = blended_img.size[1]-image_np.shape[1]
            blended_img.paste(base_img, (t_x, t_y))
    else:
        t_x = blended_img.size[0] - image_np.shape[2]
        t_y = blended_img.size[1]-image_np.shape[1]
        blended_img.paste(base_img, (t_x, t_y))


def blend_thread(i, j, x_max, image_np, valid_bit_np, ens_prediction, blended_img):
    blended_np = np.zeros((image_np.shape[1], image_np.shape[2], 3), dtype='uint8')
    helvetica = ImageFont.truetype(font="arial.ttf", size=20)
    d = ImageDraw.Draw(blended_img)
    text_color = (0, 0, 0)
    index = i * x_max + j
    base_img = np_to_pil(image_np[index][:][:][:], COLOR)
    prediction_np = ens_prediction[0]
    if max(ens_prediction[2]) - min(ens_prediction[2]) != 0 :
        var_norm = (ens_prediction[2][index] - min(ens_prediction[2]) ) / (max(ens_prediction[2]) - min(ens_prediction[2]))
    else:
        var_norm = ens_prediction[2][index]
    var_mean = ((1-var_norm) + ens_prediction[0][index])/ 2

    if valid_bit_np[index] == 1:
        if prediction_np[index] == 0:  # AC
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  # red
                    blended_np[k][z][1] = 0
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "Std: {:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 40)
            d.text(location, "AC mean: {:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 70)
            d.text(location, "Norm std: {:.2%}".format(var_norm) , font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 100)
            d.text(location, "H mean: {:.2}".format(ens_prediction[3][index]) , font=helvetica, fill=text_color)
        elif prediction_np[index] == 1:  # AD
            for k in range(0, image_np.shape[1]):
                for z in range(0, image_np.shape[2]):
                    blended_np[k][z][0] = 255  #yellow
                    blended_np[k][z][1] = 255
                    blended_np[k][z][2] = 0
            mask = np_to_pil(blended_np, COLOR)
            blended_img.paste(blend(base_img, mask, var_mean), (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "Std: {:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 50)
            d.text(location, "AD mean: {:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 90)
            d.text(location, "Norm std: {:.2%}".format(var_norm) , font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 130)
            d.text(location, "H mean: {:.2}".format(ens_prediction[3][index]) , font=helvetica, fill=text_color)
        else:
            blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 10)
            d.text(location, "Std: {:.2%}".format(ens_prediction[2][index]), font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 50)
            d.text(location, "H mean: {:.2}".format(ens_prediction[1][index]) , font=helvetica, fill=text_color)
            location = (j * image_np.shape[2] + 10 , i * image_np.shape[1]+ 90)
            d.text(location, "Norm std: {:.2%}".format(var_norm) , font=helvetica, fill=text_color)
    else:
        blended_img.paste(base_img, (j * image_np.shape[2], i * image_np.shape[1]))

"""