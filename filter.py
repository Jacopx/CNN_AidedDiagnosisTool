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

FILTER_FOLDER = path.join("resources", "filtered")


def filter_np_rgb_to_grayscale(np_image):
    start_time = time.time()
    weights = np.array([0.2125, 0.7154, 0.0721])
    np_grayscale = np.dot(np_image, weights)
    elapsed_time = time.time() - start_time
    log.print_debug("RGB converted to Gray scale || Shape: " + str(np_grayscale.shape) + "+ || Time Elapsed: " + str(elapsed_time))
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
    start_time = time.time()
    np_binary = np.copy(np_image)
    np_binary[np_binary > 100] = 255
    np_binary[np_binary <= 100] = 0
    elapsed_time = time.time() - start_time
    log.print_debug("Threshold applied || Shape: " + str(np_binary.shape) + "+ || Time Elapsed: " + str(elapsed_time))
    return np_binary


def apply_mask(np_rgb_image, np_mask):
    start_time = time.time()
    np_rgb_masked = np.zeros(np_rgb_image.shape)
    np_rgb_masked[:, :, 0] = np_rgb_image[:, :, 0] * (np_mask/255)
    np_rgb_masked[:, :, 1] = np_rgb_image[:, :, 1] * (np_mask/255)
    np_rgb_masked[:, :, 2] = np_rgb_image[:, :, 2] * (np_mask/255)
    elapsed_time = time.time() - start_time
    log.print_debug("Mask applied || Time Elapsed: " + str(elapsed_time))
    return np_rgb_masked
