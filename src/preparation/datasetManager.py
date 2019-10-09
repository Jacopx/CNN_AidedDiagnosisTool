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
RANDOM_STATE = 10
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


# Open the whole cropped dataset
# Parameters : *sampSizes -> list of sizes of samples
# Return : x -> dataset of samples, y -> related labels, p -> related patient
def load_datasets(*sampSizes):
    x = []
    y = []
    p = []
    for ss in sampSizes:
        log.print_debug("Opening Cropped dataset " + str(ss))
        cropped_dataset_folder = path.join(CROP_FOLDER,str(ss))
        for filename in os.listdir(cropped_dataset_folder):
            try :
                img_path = path.join(cropped_dataset_folder, filename)
                img_patient = filename.split("_")[0]
                img_class = CATEGORIES.index(str(filename.split("_")[1]))
                img = Image.open(img_path).convert('RGB')
                img_array = np.asarray(img, np.uint8)
                if filter.check_valid(img_array):
                    x.append(img_array)
                    y.append(img_class)
                    p.append(img_patient)
                else:
                    log.print_warning("Img "+filename+" not loaded: too much white")
            except Exception as e:
                log.print_error("Cannot load image "+filename)
    #x , y, p = balance_dataset(x, y, p)
    #x_data = np.asarray(x)
    #y_data = np.asarray(y)
    #p_data = np.asarray(p)
    #print_stats(y_data)
    return x, y, p


# DEPRECATED : Balance_dataset without taking in account patients
@DeprecationWarning
def balance_dataset(x, y, p):
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
                    img_patient = os.path.basename(filename).split("_")[0]
                    img_class = unique[i]
                    img = Image.open(filename).convert('RGB')
                    img_array = np.asarray(img, np.uint8)
                    if filter.check_valid(img_array) :
                        x.append(img_array)
                        y.append(img_class)
                        p.append(img_patient)
                    else :
                        log.print_warning("Img " + filename + " not loaded: too much white")
                        j = j - 1
                else :
                    log.print_warning("No more available images for class " + CATEGORIES[unique[i]])
                    break
    return x, y, p


# Print stats of argument dataset
# Parameters : y_data -> list of labels
def print_stats(y_data):
    unique, counts = np.unique(y_data, return_counts=True)
    for i in range(0, unique.shape[0]):
        temp = int(counts[i]/len(y_data)*100)
        log.print_info(CATEGORIES[unique[i]]+" ("+str(unique[i])+") -> "+str(counts[i])+" ("+ str(temp) +"%)")
    log.print_info("Total: "+ str(np.sum(counts)))


# Split dataset granting crops of same patient are in the same set
# Parameters : x -> dataset of samples, y -> related labels, p -> related patient, test_factor -> size rate of test set,  random_state -> seed for random generator
# Return : X_train, X_test, y_train, y_test -> split sets
def dataset_split(x, y, p, test_factor = 0.5, random_state = None):
    log.print_debug("Splitting dataset")
    dataset = []
    for i in range(0, len(y)):
        dataset.append([x[i], y[i], p[i]])
    random.seed(random_state)
    r = {p_e: random.random() for x_e, y_e, p_e in dataset}
    dataset.sort(key=lambda item: r[item[2]])
    train_size = int(len(dataset) - (len(dataset) * test_factor))

    before_different = train_size - 2
    after_different = train_size
    while dataset[before_different][2] == dataset[train_size - 1][2] :
        before_different = before_different - 1
    while dataset[after_different][2] == dataset[train_size - 1][2]:
        after_different = after_different + 1
    if train_size - before_different < after_different - train_size :
        X_train = np.asarray(dataset)[:before_different + 1, 0]
        y_train = np.asarray(dataset)[:before_different + 1, 1]
        X_test = np.asarray(dataset)[before_different + 1:, 0]
        y_test = np.asarray(dataset)[before_different + 1:, 1]
        in_train_patients = np.unique(np.asarray(dataset)[:before_different + 1, 2])
        in_test_patients = np.unique(np.asarray(dataset)[before_different + 1:, 2])
    else:
        X_train = np.asarray(dataset)[:after_different + 1, 0]
        y_train = np.asarray(dataset)[:after_different + 1, 1]
        X_test = np.asarray(dataset)[after_different + 1:, 0]
        y_test = np.asarray(dataset)[after_different + 1:, 1]
        in_train_patients = np.unique(np.asarray(dataset)[:after_different + 1, 2])
        in_test_patients = np.unique(np.asarray(dataset)[after_different + 1:, 2])

    log.print_info(" Dataset shape : " + str(X_train.shape) + " " + str(y_train.shape) + str(X_test.shape) + " " + str(y_test.shape) )

    X_train, y_train = balance_set(X_train, y_train, in_train_patients)
    X_test, y_test = balance_set(X_test, y_test, in_test_patients)

    return X_train, X_test, y_train, y_test


# Balance a set granting added crops are not in the other set
# Parameters : x -> dataset of samples, y -> related labels, in_set_patients -> patients already in the set
# Return : x, y -> balanced sets
def balance_set(x, y, in_set_patients):
    log.print_debug("Balancing dataset")
    cropped_dataset_folder = path.join(CROP_FOLDER, str(1120))
    x_list = x.tolist()
    unique, counts = np.unique(y, return_counts=True)
    max_class_size = max(counts)
    for i in range(0, unique.shape[0]):
        if counts[i] < max_class_size:
            file_list = glob.glob(path.join(cropped_dataset_folder, "*" + CATEGORIES[unique[i]] + "*"))
            cleaned_file_list = file_list.copy()
            for filename in cleaned_file_list:
                img_patient = os.path.basename(filename).split("_")[0]
                if img_patient in in_set_patients:
                    cleaned_file_list.remove(filename)
            images_to_add = max_class_size - counts[i]
            for j in range(0, max_class_size - counts[i]):
                if len(cleaned_file_list) > 0:
                    filename = random.choice(cleaned_file_list)
                    cleaned_file_list.remove(filename)
                    img_class = unique[i]
                    img = Image.open(filename).convert('RGB')
                    img_array = np.asarray(img, np.uint8)
                    if filter.check_valid(img_array):
                        # log.print_error(" X shape ----> " + str(len(x_list)))
                        x_list.append(img_array)
                        # log.print_error(" X shape ----> " + str(len(x_list)))
                        y = np.append(y, img_class)
                        images_to_add = images_to_add - 1
                        log.print_debug("Img " + filename + " added to set. " + str( images_to_add ) + " images to go.")
                    else:
                        log.print_warning("Img " + filename + " not loaded: too much white")
                        continue
                else:
                    log.print_warning("No more available images for class " + CATEGORIES[unique[i]])
                    break
    return np.asarray(x_list), y


# If already exist, open pickles containing the whole dataset. Create dataset and save it in pickles, otherwise.
# Return : X_train, X_test, y_train, y_test -> split balanced sets
def open_dataset():
    x_path = path.join(SET_FOLDER, "X.pickle")
    y_path = path.join(SET_FOLDER, "y.pickle")
    p_path = path.join(SET_FOLDER, "p.pickle")

    if os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(p_path):
        log.print_debug("Opening saved sets")
        pickle_in = open(x_path, "rb")
        X = pickle.load(pickle_in)
        pickle_in = open(y_path, "rb")
        y = pickle.load(pickle_in)
        pickle_in = open(p_path, "rb")
        p = pickle.load(pickle_in)
    else:
        X, y, p = load_datasets(1344, 2240, 3136)
        log.print_debug("Saving and opening sets")
        pickle_out = open(x_path, "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()
        pickle_out = open(y_path, "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        pickle_out = open(p_path, "wb")
        pickle.dump(p, pickle_out)
        pickle_out.close()

    log.print_info(" Dataset shape : " + str(len(X)) + " " + str(len(y)) + " " + str(len(p)))

    x_train_path = path.join(SET_FOLDER, str(RANDOM_STATE), "X_train.pickle")
    y_train_path = path.join(SET_FOLDER, str(RANDOM_STATE), "y_train.pickle")
    x_test_path = path.join(SET_FOLDER, str(RANDOM_STATE), "X_test.pickle")
    y_test_path = path.join(SET_FOLDER, str(RANDOM_STATE), "y_test.pickle")
    if os.path.isfile(x_train_path) and os.path.isfile(y_train_path) and os.path.isfile(x_test_path) and os.path.isfile(
            y_test_path):
        pickle_in = open(x_train_path, "rb")
        X_train = pickle.load(pickle_in)
        pickle_in = open(y_train_path, "rb")
        y_train = pickle.load(pickle_in)
        pickle_in = open(x_test_path, "rb")
        X_test = pickle.load(pickle_in)
        pickle_in = open(y_test_path, "rb")
        y_test = pickle.load(pickle_in)
    else:
        X_train, X_test, y_train, y_test = dataset_split(X, y, p, test_factor=TEST_SIZE, random_state=RANDOM_STATE)
        pickle_out = open(x_train_path, "wb")
        pickle.dump(X_train, pickle_out)
        pickle_out.close()
        pickle_out = open(y_train_path, "wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
        pickle_out = open(x_test_path, "wb")
        pickle.dump(X_test, pickle_out)
        pickle_out.close()
        pickle_out = open(y_test_path, "wb")
        pickle.dump(y_test, pickle_out)
        pickle_out.close()

    return X_train , y_train, X_test, y_test