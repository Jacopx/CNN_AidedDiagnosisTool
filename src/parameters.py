from os import path

###############################################################

INPUT_SIZE = 64
TEST_SIZE = 0.25
RANDOM_STATE = 10
CATEGORIES = ["AC","AD","H"]
#CROP_SIZE = 4480

N_CLASSES = 3
N_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DECAY = 1e-6
#DROP_RATE = 0.5
USE_BIAS = True
DENSE_SIZE = 2048
SEPARABLE_CONVOLUTION = False

DATASET_FOLDER = path.join("resources", "DATASET_ROI")
MAP_FOLDER = path.join("resources", "DATASET_ROI", "MAP")
FILTER_FOLDER = path.join("resources", "filtered")
RESOURCE_FOLDER = path.join("resources")
CROP_FOLDER = path.join("resources", "cropped_dataset","overlap_no_padding")
SET_FOLDER = path.join("resources", "cropped_dataset","dataset_train_test",str(INPUT_SIZE))
DUMMY_SET_FOLDER = path.join("resources", "cropped_dataset","dummyset_train_test",str(INPUT_SIZE))
MODEL_FOLDER = path.join("resources", "models")
SUMMARY_FOLDER = path.join("resources", "models","summary")
PREDICTION_FOLDER = path.join("resources", "predictions")


###############################################################


