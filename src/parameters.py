from os import path
import arguments

###############################################################

INPUT_SIZE = 32
TEST_SIZE = 0.25
RANDOM_STATE = 10
CATEGORIES = ["AC","AD","H"]

DATASET_FOLDER = arguments.get_dataset_folder()
FILTER_FOLDER = path.join("resources", "filtered")
CROP_FOLDER = path.join("resources", "cropped_dataset","overlap_no_padding")
SET_FOLDER = path.join("resources", "cropped_dataset","dataset_train_test",str(INPUT_SIZE))
DUMMY_SET_FOLDER = path.join("resources", "cropped_dataset","dummyset_train_test",str(INPUT_SIZE))
MODEL_FOLDER = path.join("resources", "models")
SUMMARY_FOLDER = path.join("resources", "models","summary")

N_CLASSES = 3
N_EPOCH = 1
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
DECAY = 1e-6
DROP_RATE = 0
USE_BIAS = False
DENSE_SIZE = 1024
SEPARABLE_CONVOLUTION = False

###############################################################