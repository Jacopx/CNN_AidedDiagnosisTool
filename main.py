# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#            usage: main.py [-h] [-v] dataset_folder sample_size            *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import sys
sys.path.append("..")
import logger as log
import time
import src.cnn.testCNN as convNet
from src.preparation import datasetManager as dm, utils
from src.parameters import *
import gc


def train():
    X_train, y_train, X_test, y_test = dm.open_dataset()
    log.print_info(" TRAIN STATs")
    log.print_info(" Train set shape : " + str(X_train.shape) + " " + str(y_train.shape))
    log.print_info(" Train set type : " + str(X_train.dtype))
    dm.print_stats(y_train)
    log.print_info(" TEST STATs")
    log.print_info(" Test set shape : " + str(X_test.shape) + " " + str(y_test.shape))
    log.print_info(" Test set type : " + str(X_test.dtype))
    dm.print_stats(y_test)
    convNet.compile_model(X_train, y_train, X_test, y_test)


def produce_test_predictions():
    iter = [1, 10]
    drop_rates = ["0.01", "0.1", "0.5"]
    for iterations in iter:
        for dr in drop_rates:
            pred_folder = path.join("resources", "predictions", dr)
            for i in range(1, 6):
                start_time = time.time()
                log.print_debug("PROCESSING map_" + str(i) + ".svs dr:" + str(dr) + " iter:" + str(iterations)+ " Crop Size:" + str(CROP_SIZE))
                setDropRate(dr)
                dm.mc_predict(iterations, "map_" + str(i) + ".svs", pred_folder, dr)
                mc_predictions = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + "_" + str(iterations) + ".pred",
                                              pred_folder)
                ens_predictions_np = dm.compute_ens_predictions(mc_predictions)
                dm.save_blob(ens_predictions_np,
                             "map_" + str(i) + "_" + str(CROP_SIZE) + "_" + str(iterations) + ".ens",
                             pred_folder)
                np_flatten_image = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + ".bin", MAP_FOLDER)
                slide_size = dm.read_blob("map_" + str(i) + ".info", MAP_FOLDER)
                valid_bit_np = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + ".vbit", MAP_FOLDER)
                if iterations == 1:
                    image = utils.blend_np_gradient(np_flatten_image, ens_predictions_np, valid_bit_np, slide_size,
                                                    False)
                else:
                    image = utils.blend_np_gradient(np_flatten_image, ens_predictions_np, valid_bit_np, slide_size,
                                                    True)
                utils.save_image(image, RESOURCE_FOLDER,
                                 "map_" + str(i) + "_CS" + str(CROP_SIZE) + "_DR" + dr.replace("0.",
                                                                                               "") + "_ITER" + str(
                                     iterations) + "_gradient")
                elapsed_time = time.time() - start_time
                log.print_debug(
                    "map_" + str(i) + "_CS" + str(CROP_SIZE) + "_DR" + dr.replace("0.", "") + "_ITER" + str(
                        iterations) + "_gradient" + ": " + str(elapsed_time))
                gc.collect()


def produce_gradient_images():
    iter = [1, 10, 100]
    drop_rates = ["0.01", "0.1", "0.5"]
    for iterations in iter:
        for dr in drop_rates:
            pred_folder = path.join("resources", "predictions", dr)
            for i in range(1, 6):
                start_time = time.time()
                log.print_debug("PROCESSING map_" + str(i) + ".svs dr:"+ str(dr) + " iter:" + str(iterations))
                ens_predictions_np = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + "_" + str(iterations) + ".ens", pred_folder)
                np_flatten_image = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + ".bin", MAP_FOLDER)
                slide_size = dm.read_blob("map_" + str(i) + ".info", MAP_FOLDER)
                valid_bit_np = dm.read_blob("map_" + str(i) + "_" + str(CROP_SIZE) + ".vbit", MAP_FOLDER)
                if iterations == 1:
                    image = utils.blend_np_gradient(np_flatten_image, ens_predictions_np, valid_bit_np, slide_size, False)
                else:
                    image = utils.blend_np_gradient(np_flatten_image, ens_predictions_np, valid_bit_np, slide_size, True)
                utils.save_image(image, RESOURCE_FOLDER,
                                 "map_" + str(i) + "_CS" + str(CROP_SIZE) + "_DR" + dr.replace("0.", "") + "_ITER" + str(iterations) + "_gradient")
                elapsed_time = time.time() - start_time
                log.print_debug(
                    "map_" + str(i) + "_CS" + str(CROP_SIZE) + "_DR" + dr.replace("0.", "") + "_ITER" + str(iterations) + "_gradient" + ": " + str(elapsed_time))
                gc.collect()

def main():
    produce_test_predictions()
    return 0


if __name__ == "__main__":
   main()
