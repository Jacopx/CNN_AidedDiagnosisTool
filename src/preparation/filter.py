# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *       slide.py : tools for images manipulation filtering and            *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import numpy as np
import time
import logger as log
from os import path
#import skimage.filters as sk_filters
import cv2 as cv

FILTER_FOLDER = path.join("resources", "filtered")


def filter_np_rgb_to_grayscale(np_image):
    start_time = time.time()
    weights = np.array([0.2125, 0.7154, 0.0721])
    np_grayscale = np.dot(np_image, weights)
    elapsed_time = time.time() - start_time
    #log.print_debug("RGB converted to Gray scale || Shape: " + str(np_grayscale.shape) + "+ || Time Elapsed: " + str(elapsed_time))
    return np_grayscale


def filter_np_rgb_to_cv_grayscale(np_image):
    start_time = time.time()
    np_grayscale = cv.cvtColor(np.asarray(np_image), cv.COLOR_RGB2GRAY)
    elapsed_time = time.time() - start_time
    log.print_debug("RGB converted to CV Gray scale || Shape: " + str(np_grayscale.shape) + "+ || Time Elapsed: " + str(elapsed_time))
    return np_grayscale


def filter_rgb_to_grayscale(image):
    start_time = time.time()
    image_grayscale = image.convert(mode="L")
    elapsed_time = time.time() - start_time
    log.print_debug("RGB converted to Gray scale || Time Elapsed: " + str(elapsed_time))
    return image_grayscale


def complement_np(np_image):
    start_time = time.time()
    np_complemented = 255 - np_image
    elapsed_time = time.time() - start_time
    log.print_debug("Image complemented || Shape: " + str(np_complemented.shape) + "+ || Time Elapsed: " + str(elapsed_time))
    return np_complemented


def filter_np_threshold(np_image):
    np_binary = np.copy(np_image)
    np_binary[np_binary > 142] = 255
    np_binary[np_binary <= 142] = 0
    return np_binary


def check_valid(np_image):
    valid = True
    start_time = time.time()
    np_grayscale = filter_np_rgb_to_grayscale(np_image)
    np_binary = np.copy(np_grayscale)
    np_binary[np_binary > 230] = 255
    np_binary[np_binary <= 230] = 0
    black_px = sum(sum(i == 0 for i in np_binary))
    white_px = sum(sum(i == 255 for i in np_binary))
    if white_px/(white_px + black_px) > 0.7:
        valid = False
    elapsed_time = time.time() - start_time
    #log.print_debug("Threshold applied || Shape: " + str(np_binary.shape) + "+ || Time Elapsed: " + str(elapsed_time))
    return valid


def apply_mask(np_rgb_image, np_mask):
    start_time = time.time()
    np_rgb_masked = np.zeros(np_rgb_image.shape)
    np_rgb_masked[:, :, 0] = np_rgb_image[:, :, 0] * (np_mask/255)
    np_rgb_masked[:, :, 1] = np_rgb_image[:, :, 1] * (np_mask/255)
    np_rgb_masked[:, :, 2] = np_rgb_image[:, :, 2] * (np_mask/255)
    elapsed_time = time.time() - start_time
    log.print_debug("Mask applied || Time Elapsed: " + str(elapsed_time))
    return np_rgb_masked

"""
def otsu_filter(np_gs_image):
    start_time = time.time()
    otsu_thresh_value = sk_filters.threshold_otsu(np_gs_image)
    log.print_error(otsu_thresh_value)
    np_otsu = np.copy(np_gs_image)
    np_otsu[np_otsu > otsu_thresh_value] = 255
    np_otsu[np_otsu <= otsu_thresh_value] = 0
    elapsed_time = time.time() - start_time
    log.print_debug("Otsu filter scikit || Time Elapsed: " + str(elapsed_time))
    return np_otsu


def gaussian_filter(np_gs_image, s, t):
    start_time = time.time()
    np_gaussian = sk_filters.gaussian(np_gs_image, sigma=s, truncate=t, multichannel=False)
    elapsed_time = time.time() - start_time
    log.print_debug("Otsu filter scikit || Time Elapsed: " + str(elapsed_time))
    return np_gaussian
"""

def normalize_filter(np_image):
    start_time = time.time()
    np_image *= 255.0 / np_image.max()
    elapsed_time = time.time() - start_time
    log.print_debug("Otsu filter scikit || Time Elapsed: " + str(elapsed_time))
    return np_image

