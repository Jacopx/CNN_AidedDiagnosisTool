# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os
import sys
import numpy as np
import utils
import filter
import time
import logger as log

DATASET_FOLDER = sys.argv[1]


def min_max_ss():
    size_list = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            w_temp, h_temp = slide.get_slidepath_size(DATASET_FOLDER + "/" + filename)
            size_list.append(w_temp)
            size_list.append(h_temp)
    return min(size_list)


def produce_crops(folder):
    # Production
    ss = min_max_ss()
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide.overlap_crop_multithread(DATASET_FOLDER, filename, ss)


def main():
    start_time = time.time()
    s = slide.open_slide(DATASET_FOLDER+"\\2_AC_1.svs")
    image = slide.slide_to_image(s)
    image_rgb = image.convert('RGB')
    image_resized = slide.resize_image_r(image_rgb, slide.SCALE_FACTOR)
    np_rgb = utils.image_to_np_rgb(image_resized)
    np_grayscale = filter.filter_np_rgb_to_grayscale(np_rgb)
    np_grayscale_complemented = filter.complement_np(np_grayscale)
    np_complemented_binary = filter.filter_np_threshold(np_grayscale_complemented)
    image_grayscale_complemented = utils.np_to_pil(np_grayscale_complemented, utils.GRAY_SCALE)
    image_complemented_binary = utils.np_to_pil(np_complemented_binary, utils.GRAY_SCALE)
    utils.save_image(image_resized, "filter", "temp_rgb.png")
    utils.save_image(image_grayscale_complemented, "filter", "temp_gs_c.png")
    utils.save_image(image_complemented_binary, "filter", "temp_bw_c.png")
    elapsed_time = time.time() - start_time
    log.print_debug("TOTAL TIME FOR IMAGE PROCESSING: " + str(elapsed_time))


if __name__ == "__main__":
    main()
