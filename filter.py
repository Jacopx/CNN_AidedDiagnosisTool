# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

from PIL import  Image
import numpy as np


def filter_np_rgb_to_grayscale(np_image):
    np_grayscale = np.empty([np_image.shape[0], np_image.shape[1]], dtype="uint8")
    for i in range(0, np_image.shape[0]):
        for j in range(0, np_image.shape[1]):
            np_grayscale[i][j] = (np_image[i][j][0]*0.2125 + np_image[i][j][1]*0.7154 + np_image[i][j][2]*0.0721).astype('uint8')
    return np_grayscale


def filter_rgb_to_grayscale(image):
    return image.convert(mode="L")


def complement_np(np_image):
    np_complemented = np.empty([np_image.shape[0], np_image.shape[1]], dtype="uint8")
    for i in range(0, np_image.shape[0]):
        for j in range(0, np_image.shape[1]):
            np_complemented[i][j] = (255 - np_image[i][j]).astype('uint8')
    return np_complemented

