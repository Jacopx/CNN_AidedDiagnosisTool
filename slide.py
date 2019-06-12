# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide
import time
import math
from threading import Thread
import utils
import logger as log
import sys
from os import path


LVL = 0  # Level of dimensionality
CROP_FOLDER = path.join("resources", "cropped_dataset")
SCALE_FACTOR = 24


def open_slide(slide_path):
    start_time = time.time()
    slide = openslide.open_slide(slide_path)
    elapsed_time = time.time() - start_time
    log.print_debug("Opened slide " + slide_path + " || Shape: " + str(get_slide_size(slide)) + " || Time Elapsed: " + str(elapsed_time))
    return slide


def slide_to_image(slide):
    start_time = time.time()
    log.print_debug("Converting slide to image. Requires time!")
    width, height = get_slide_size(slide)
    # MAX width*height = 2**29!!!!!!! IMPORTANT
    image = slide.read_region((0, 0), LVL, (width, height))
    elapsed_time = time.time() - start_time
    log.print_debug("Converted slide to image || Shape: " + str(image.size) + "+ || Time Elapsed: " + str(elapsed_time))
    return image


def resize_image_r(image, scale_factor):
    start_time = time.time()
    width, height = image.size
    image_r = image.resize((int(width / scale_factor), int(height / scale_factor)))
    elapsed_time = time.time() - start_time
    log.print_debug("Image resized || Shape: " + str(image.size) + "+ || Time Elapsed: " + str(elapsed_time))
    return image_r


def resize_image_a(image, width, height):
    start_time = time.time()
    image_r = image.resize((int(width), int(height)))
    elapsed_time = time.time() - start_time
    log.print_debug("Image resized || Time Elapsed: " + str(elapsed_time))
    return image_r


def slide_info(slide_path):
    print("Whole-slide info ", end="")
    slide = open_slide(slide_path)
    width, height = get_slide_size(slide)
    tile = slide.properties['openslide.level[0].tile-width']
    print(slide_path)
    print("\tDimension: " + str(width) + "x" + str(height))
    print("\tTile: " + str(tile))
    print("\tProperties: " + str(slide.properties))
    print("\tLevel dimension: " + str(slide.level_dimensions))


def get_slidepath_size(slide_path):
    if slide_path is not None:
        slide = open_slide(slide_path)

    if slide is not None:
        width = int(slide.dimensions[0])
        height = int(slide.dimensions[1])
    else:
        width = -1
        height = -1
    return width, height


def get_slide_size(slide):
    if slide is not None:
        width = int(slide.dimensions[0])
        height = int(slide.dimensions[1])
    else:
        width = -1
        height = -1
    return width, height


def make_crop_folder(algorithm, custom_ss):
    folder_path = path.join(CROP_FOLDER, algorithm, str(custom_ss))
    utils.make_folder(folder_path)


"""
def overlap_crop_singleprocess(dataset_folder, slide_name_ex, custom_ss):

    # Folder Creation (if not exist)
    algorithm_crop_folder = "overlap_no_padding"
    make_crop_folder(algorithm_crop_folder, custom_ss)

    # Slide opening
    slide_path = dataset_folder + "/" + slide_name_ex
    slide_name = slide_name_ex.split(".")[0]
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])
    print("Get image...")
    image = slide.read_region((0, 0), LVL, (width, height))
    # Computing number of windows
    w_windows = int(math.ceil(width / custom_ss))
    h_windows = int(math.ceil(height / custom_ss))
    # Computing overlap for a single window
    if (w_windows - 1) != 0:
        w_overlap = math.floor(((custom_ss * w_windows) - width) / (w_windows - 1))
    else:
        w_overlap = 0
    if (h_windows - 1) != 0:
        h_overlap = math.floor(((custom_ss * h_windows) - height) / (h_windows - 1))
    else:
        h_overlap = 0

    # Cropping
    crop_number = 0
    y = 0
    i_w = 0
    i_h = 0
    for i in range(0, h_windows - 1):
        x = 0
        for j in range(0, w_windows - 1):
            crop_number += 1
            box = (x, y, x + custom_ss, y + custom_ss)
            crop_region = image.crop(box)
            crop_name = CROP_FOLDER + "/" + algorithm_crop_folder + "/" + str(custom_ss) + "/" + slide_name + "_" + str(
                i_h) + 'x' + str(i_w) + ".png"
            print("Saving crop " + str(crop_number) + " ----> " + str(box) + " " + str(i_h) + "x" + str(i_w) + "name:" + crop_name)
            crop_region.save(crop_name)
            x += custom_ss - w_overlap
            i_w += 1
        crop_number += 1
        box = (width - custom_ss, y, width, y + custom_ss)
        crop_region = image.crop(box)
        crop_name = CROP_FOLDER + "/" + algorithm_crop_folder + "/" + str(custom_ss) + "/" + slide_name + "_" + str(
            i) + 'x' + str(j) + ".png"
        print("Saving crop " + str(crop_number) + " ----> " + str(box) + " " + str(i_h) + "x" + str(
            i_w) + "name:" + crop_name)
        crop_region.save(crop_name)
        y += custom_ss - h_overlap
        i_w = 0
        i_h += 1
    x = 0
    for j in range(0, w_windows - 1):
        crop_number += 1
        box = (x, height - custom_ss, x + custom_ss, height)
        print("Crop " + str(crop_number) + " ----> " + str(box) + " " + str(i_h) + "x" + str(i_w))
        crop_region = image.crop(box)
        crop_name = CROP_FOLDER + "/" + algorithm_crop_folder + "/" + str(custom_ss) + "/" + slide_name + "_" + str(
            i_h) + 'x' + str(i_w) + ".png"
        crop_region.save(crop_name)
        x += custom_ss - w_overlap
        i_w += 1
    box = (width - custom_ss, height - custom_ss, width, height)
    crop_region = image.crop(box)
    crop_name = CROP_FOLDER + "/" + algorithm_crop_folder + "/" + str(custom_ss) + "/" + slide_name + "_" + str(i_h) + 'x' + str(
        i_w) + ".png"
    print("Saving crop " + str(crop_number) + " ----> " + str(box) + " " + str(i_h) + "x" + str(
        i_w) + "name:" + crop_name)
    crop_region.save(crop_name)
    print("---------------------------------------------------------")
"""


def custom_crop(image, box, crop_name):
    crop_region = image.crop(box)
    crop_region.save(crop_name)

"""
def overlap_crop_multithread(dataset_folder, slide_name_ex, custom_ss):

    start_time = time.time()

    pool = []
    x_p = []
    y_p = []

    # Folder Creation (if not exist)
    algorithm_crop_folder = "overlap_no_padding"
    make_crop_folder(algorithm_crop_folder, custom_ss)

    # Slide opening
    slide_path = path.join(dataset_folder, slide_name_ex)
    slide_name = slide_name_ex.split(".")[0]
    slide = open_slide(slide_path)
    image = slide_to_image(slide)
    width, height = get_slide_size(slide)
    # Computing number of windows
    w_windows = int(math.ceil(width / custom_ss))
    h_windows = int(math.ceil(height / custom_ss))
    # Computing overlap for a single window
    if (w_windows - 1) != 0:
        w_overlap = math.floor(((custom_ss * w_windows) - width) / (w_windows - 1))
    else:
        w_overlap = 0
    if (h_windows - 1) != 0:
        h_overlap = math.floor(((custom_ss * h_windows) - height) / (h_windows - 1))
    else:
        h_overlap = 0
    x = 0
    for j in range(0, w_windows - 1):
        x_p.append(x)
        x += custom_ss - w_overlap
    y = 0
    for i in range(0, h_windows - 1):
        y_p.append(y)
        y += custom_ss - h_overlap
    x_p.append(width-custom_ss)
    y_p.append(height-custom_ss)

    # Cropping
    crop_number = 0
    for i in range(0, len(x_p)):
        for j in range(0, len(y_p)):
            crop_number += 1
            box = (x_p[i], y_p[j], x_p[i] + custom_ss, y_p[j] + custom_ss)
            crop_name = CROP_FOLDER + "/" + algorithm_crop_folder + "/" + str(custom_ss) + "/" + slide_name + "_" + str(i) + 'x' + str(j) + ".png"
            pool.append(Thread(target=custom_crop, args=(image, box, crop_name,)))
            pool[-1].start()
    for p in pool:
        p.join()
    elapsed_time = time.time() - start_time
    log.print_debug(str(crop_number+1) + " crops produced || Time Elapsed: " + str(elapsed_time))


def resize_crop(folder, file_folder, slide, custom_ss):
    start_time = time.time()
    make_crop_folder("resize", custom_ss)
    slide_path = path.join(folder, slide)
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])

    # Computing overlapping for width
    w_sample = width / custom_ss
    dec_w = w_sample % 1
    over_w = dec_w * custom_ss

    # Computing overlapping for height
    h_sample = height / custom_ss
    dec_h = h_sample % 1
    over_h = dec_h * custom_ss

    print("Get image...")
    image = slide.read_region((0, 0), LVL, (width, height))
    print("Resize image...")
    image.resize((int(width - over_w), int(height - over_h)))

    print("Starting crop...")
    # # DEBUG CYCLE
    # for shift_h in range(0, 10):
    #     for shift_w in range(0, 10):
    crop_number = 0
    for shift_w in range(0, int(w_sample)):
        for shift_h in range(0, int(h_sample)):
            if shift_h == 0 and shift_w == 0:
                box = (0, 0, custom_ss, custom_ss)
            elif shift_h == 0:
                box = (custom_ss * shift_w, 0, custom_ss * (shift_w + 1), custom_ss)
            elif shift_h == 0:
                box = (0, custom_ss * shift_h, 0, custom_ss * (shift_h + 1))
            else:
                box = (custom_ss * shift_w, custom_ss * shift_h, custom_ss * (shift_w + 1), custom_ss * (shift_h + 1))

            print(box, end=" ==> ")
            crop_region = image.crop(box)
            crop_name = path.join(CROP_FOLDER, 'resize', custom_ss, file_folder, "_", str(shift_w), "x", str(shift_h))
            crop_region.save(crop_name + ".png")
            print(str(shift_w) + 'x' + str(shift_h))
            crop_number += 1

    elapsed_time = time.time() - start_time
    print("\n # Crop: " + str(crop_number) + " || Time Elapsed: " + str(elapsed_time))
"""
