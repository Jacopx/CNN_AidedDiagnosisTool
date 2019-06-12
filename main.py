# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os
from os import path
import sys
import utils
import filter
import time
import logger as log

DATASET_FOLDER = sys.argv[1]


def min_max_ss():  # Get the greatest dimension of the dataset of training
    size_list = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide_path = path.join(DATASET_FOLDER, filename)
            w_temp, h_temp = slide.get_slidepath_size(slide_path)
            size_list.append(w_temp)
            size_list.append(h_temp)
    return min(size_list)


def produce_crops():  # Produce the crop for the training
    # Production
    ss = min_max_ss()
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide.overlap_crop_multithread(DATASET_FOLDER, filename, ss)


def main():
    start_time = time.time()
    """
    slide_path = path.join(DATASET_FOLDER, "2_AC_1.svs")
    s = slide.open_slide(slide_path)
    image = slide.slide_to_image(s)
    image_rgb = image.convert('RGB')
    image_resized = slide.resize_image_r(image_rgb, slide.SCALE_FACTOR)
    np_rgb = utils.image_to_np_rgb(image_resized)
    np_grayscale = filter.filter_np_rgb_to_grayscale(np_rgb)
    np_grayscale_complemented = filter.complement_np(np_grayscale)
    np_complemented_binary = filter.filter_np_threshold(np_grayscale_complemented)
    np_masked = filter.apply_mask(np_rgb, np_complemented_binary)
    image_grayscale_complemented = utils.np_to_pil(np_grayscale_complemented, utils.GRAY_SCALE)
    image_complemented_binary = utils.np_to_pil(np_complemented_binary, utils.GRAY_SCALE)
    image_rgb_masked = utils.np_to_pil(np_masked, utils.COLOR)
    utils.save_image(image_resized, "filter", "temp_rgb.png")
    utils.save_image(image_grayscale_complemented, "filter", "temp_gs_c.png")
    utils.save_image(image_complemented_binary, "filter", "temp_bw_c.png")
    utils.save_image(image_rgb_masked, "filter", "final_cl.png")
    """
    elapsed_time = time.time() - start_time
    log.print_debug("TOTAL TIME FOR IMAGE PROCESSING: " + str(elapsed_time))


if __name__ == "__main__":
    main()
