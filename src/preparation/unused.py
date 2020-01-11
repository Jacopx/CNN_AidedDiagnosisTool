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

