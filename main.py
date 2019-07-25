# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#                  usage: main.py [-h] [-v] dataset_folder                  *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import slide
import os
from os import path
import cv2 as cv
import utils
import filter
import time
import logger as log
import arguments


DATASET_FOLDER = arguments.get_dataset_folder()
FILTER_FOLDER = path.join("resources", "filtered")


def min_max_ss():  # Get the greatest dimension of the dataset of training
    size_list = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide_path = path.join(DATASET_FOLDER, filename)
            w_temp, h_temp = slide.get_slidepath_size(slide_path)
            size_list.append(w_temp)
            size_list.append(h_temp)
    return min(size_list)


def produce_crops(ss):  # Produce the crop for the training
    # Production
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide.overlap_crop_multithread(DATASET_FOLDER, filename, ss)


# *********************************************************************
# VVVVVVVVVVVVVVVVVVVVV IMPOSSIBLE TO BE EXECUTED VVVVVVVVVVVVVVVVVVVVV
def produce_filtered_slides():
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide_path = path.join(DATASET_FOLDER, filename)
            s = slide.open_slide(slide_path)
            image = slide.slide_to_image(s)
            image_rgb = image.convert('RGB')
            np_rgb = utils.image_to_np_rgb(image_rgb)
            np_grayscale = filter.filter_np_rgb_to_grayscale(np_rgb)
            np_grayscale_complemented = filter.complement_np(np_grayscale)
            np_complemented_binary = filter.filter_np_threshold(np_grayscale_complemented)
            np_masked = filter.apply_mask(np_rgb, np_complemented_binary)
            image_rgb_masked = utils.np_to_pil(np_masked, utils.COLOR)
            utils.save_image(image_rgb_masked, FILTER_FOLDER, filename+"_f")


def produce_otsu_slides():
    slide_path_array = []
    # slide_path_array.append("7_H_2")
    # slide_path_array.append("7_H_4")
    # slide_path_array.append("44_H_2")
    slide_path_array.append("44_H_2")
    slide_path_array.append("18_H_3")
    for slide_name in slide_path_array:
        slide_path = path.join(DATASET_FOLDER, slide_name+".svs")
        s = slide.open_slide(slide_path)
        image = slide.slide_to_image(s)
        image_rgb = image.convert('RGB')
        image_resized = slide.resize_image_r(image_rgb, 10)
        image_resized.show()
        np_rgb = utils.image_to_np_rgb(image_resized)
        np_grayscale = filter.filter_np_rgb_to_grayscale(np_rgb)
        np_normalized = filter.normalize_filter(np_grayscale) #NO
        """grayscale -> gaussian"""
        for s in range(0, 11):
            #np_grayscale = filter.filter_np_rgb_to_grayscale(filter.gaussian_filter(np_rgb, s/10, 4))
            np_otsu_sk = filter.otsu_filter(np_normalized)
            np_otsu_sk_complemented = filter.complement_np(np_otsu_sk)
            utils.np_to_pil(filter.apply_mask(np_rgb, np_otsu_sk_complemented), utils.COLOR).show()

        """ grayscale->otsu->complement """
        """np_otsu_sk = filter.otsu_filter_scikit(np_grayscale)
        np_otsu_sk_complemented = filter.complement_np(np_otsu_sk)
        utils.np_to_pil(np_otsu_sk_complemented, utils.GRAY_SCALE).show()"""
        """ grayscale->complement->otsu"""


def main():
    start_time = time.time()
    produce_otsu_slides()
    elapsed_time = time.time() - start_time
    log.print_debug("TOTAL TIME FOR IMAGE PROCESSING: " + str(elapsed_time))


if __name__ == "__main__":
    main()