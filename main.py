# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#            usage: main.py [-h] [-v] dataset_folder sample_size            *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import os
from os import path
import glob
import random
from src.preparation import filter, slide, utils
import time
import logger as log
import arguments
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


DATASET_FOLDER = arguments.get_dataset_folder()
SAMPLE_SIZE = 1344
TEST_SIZE = 0.25
RANDOM_STATE = 0
FILTER_FOLDER = path.join("resources", "filtered")
CROP_FOLDER = path.join("resources", "cropped_dataset","overlap_no_padding")
SET_FOLDER = path.join("resources", "cropped_dataset","dataset_train_test")
CATEGORIES = ["AC","AD","H"]


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
            utils.save_image(image_rgb_masked, FILTER_FOLDER, filename + "_f")


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


def load_datasets(*sampSizes):
    x = []
    y = []
    for ss in sampSizes:
        log.print_debug("Opening Cropped dataset " + str(ss))
        cropped_dataset_folder = path.join(CROP_FOLDER,str(ss))
        for filename in os.listdir(cropped_dataset_folder):
            try :
                img_path = path.join(cropped_dataset_folder, filename)
                img_class = CATEGORIES.index(str(filename.split("_")[1]))
                img = Image.open(img_path).convert('RGB')
                img_array = np.asarray(img, np.uint8)
                if filter.check_valid(img_array):
                    x.append(img_array)
                    y.append(img_class)
                else:
                    log.print_warning("Img "+filename+" not loaded: too much white")
            except Exception as e:
                log.print_error("Cannot load image "+filename)
    x , y = balance_dataset(x, y)
    x_data = np.asarray(x)
    y_data = np.asarray(y)
    #print_stats(y_data)
    return x_data, y_data


def balance_dataset(x, y):
    cropped_dataset_folder = path.join(CROP_FOLDER,str(1120))
    unique, counts = np.unique(y, return_counts=True)
    max_class_size = max(counts)
    for i in range(0, unique.shape[0]):
        if counts[i] < max_class_size :
            fileList = glob.glob(path.join(cropped_dataset_folder ,"*"+CATEGORIES[unique[i]]+"*"))
            for j in range(0, max_class_size-counts[i]):
                if len(fileList) > 0 :
                    filename = random.choice(fileList)
                    fileList.remove(filename)
                    img_class = unique[i]
                    img = Image.open(filename).convert('RGB')
                    img_array = np.asarray(img, np.uint8)
                    if filter.check_valid(img_array) :
                        x.append(img_array)
                        y.append(img_class)
                    else :
                        log.print_warning("Img " + filename + " not loaded: too much white")
                        j = j - 1
                else :
                    log.print_warning("No more available images for class " + CATEGORIES[unique[i]])
                    break
    return x, y


def print_stats(y_data):
    unique, counts = np.unique(y_data, return_counts=True)
    for i in range(0, unique.shape[0]):
        temp = int(counts[i]/y_data.shape[0]*100)
        log.print_info(CATEGORIES[unique[i]]+" ("+str(unique[i])+") -> "+str(counts[i])+" ("+ str(temp) +"%)")
    log.print_info("Total: "+ str(np.sum(counts)))


def main():
    start_time = time.time()
    #produce_crops(3136)
    x_path = path.join(SET_FOLDER,"X.pickle")
    y_path = path.join(SET_FOLDER,"y.pickle")

    if os.path.isfile(x_path) and os.path.isfile(y_path) :
        pickle_in = open(x_path, "rb")
        X = pickle.load(pickle_in)
        pickle_in = open(y_path, "rb")
        y = pickle.load(pickle_in)
    else :
        X, y = load_datasets(1344, 2240, 3136)
        pickle_out = open(x_path, "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()
        pickle_out = open(y_path, "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_STATE)
    log.print_info("TRAIN STATs")
    print_stats(y_train)
    log.print_info("TEST STATs")
    print_stats(y_test)
    elapsed_time = time.time() - start_time
    log.print_debug("TOTAL TIME FOR PROCESSING: " + str(elapsed_time))


if __name__ == "__main__":
    main()