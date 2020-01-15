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
from os import path
import src.preparation.blender as blnd


# Train the network
# Parameters :  drop_rate        -> dropout rate
def train(drop_rate):
    X_train, y_train, X_test, y_test = dm.open_dataset()
    log.print_info(" TRAIN STATs")
    log.print_info(" Train set shape : " + str(X_train.shape) + " " + str(y_train.shape))
    log.print_info(" Train set type : " + str(X_train.dtype))
    dm.print_stats(y_train)
    log.print_info(" TEST STATs")
    log.print_info(" Test set shape : " + str(X_test.shape) + " " + str(y_test.shape))
    log.print_info(" Test set type : " + str(X_test.dtype))
    dm.print_stats(y_test)
    convNet.compile_model(X_train, y_train, X_test, y_test, drop_rate)

# Compute predictions according to parameters
# Parameters :  path_list -> a list of path
#               crop_size -> the size of crops in which the image is cut
#               dr        -> dropout rate
#               iterations-> number of iteration for MCDropout
# Return : the path of the masked image and the path of the original image
def make_prediction(path_list, crop_size, dr, iterations):
    prediction_images_path = []
    for file in path_list:
        if path.isfile(file):
            start_time = time.time()
            log.print_debug("PROCESSING "+ file +" dr:" + str(dr) + " iter:" + str(iterations) + " Crop Size" + str(crop_size))
            basename = str(path.basename(file).split(".", 1)[0])
            pred_folder = path.join("resources", "predictions", str(dr))
            suffix = "_CS" + str(crop_size) + "_DR" + str(dr).replace("0.", "") + "_ITER" + str(iterations)
            filename_std_bi      = basename + suffix + "_std_bigradient"
            filename = basename + "_CS" + str(crop_size)

            if not path.isfile(path.join(pred_folder, basename + "_" + str(crop_size) + "_" + str(iterations) + ".pred")):
                dm.mc_predict_from_path(iterations, file, pred_folder, dr, crop_size)
            mc_predictions = dm.read_blob(basename + "_" + str(crop_size) + "_" + str(iterations) + ".pred", pred_folder)
            if not path.isfile(path.join(pred_folder, basename + "_" + str(crop_size) + "_" + str(iterations) + ".ens")):
                ens_predictions_np = dm.compute_ens_predictions(mc_predictions)
                dm.save_blob(ens_predictions_np, basename + "_" + str(crop_size) + "_" + str(iterations) + ".ens",
                             pred_folder)
            ens_predictions_np = dm.read_blob(basename + "_" + str(crop_size) + "_" + str(iterations) + ".ens",
                             pred_folder)
            np_flatten_image = dm.read_blob(basename + "_" + str(crop_size) + ".bin", MAP_FOLDER)
            slide_size = dm.read_blob(basename + ".info", MAP_FOLDER)
            valid_bit_np = dm.read_blob(basename + "_" + str(crop_size) + ".vbit", MAP_FOLDER)

            if not path.isfile(path.join(SEGMENTED_FOLDER, filename_std_bi + ".png")):
                if iterations == 1:
                    image = blnd.blend_np_std_bigradient(np_flatten_image, ens_predictions_np, valid_bit_np,
                                                             slide_size, False, crop_size)
                else:
                    image = blnd.blend_np_std_bigradient(np_flatten_image, ens_predictions_np, valid_bit_np,
                                                             slide_size, True, crop_size)
                utils.save_image(image, SEGMENTED_FOLDER, filename_std_bi)

            if not path.isfile(path.join(SEGMENTED_FOLDER, filename + ".png")) and path.isfile(path.join(MAP_FOLDER, basename + "_" + str(crop_size) + ".bin")):
                if np_flatten_image is None:
                    np_flatten_image = dm.read_blob(basename + "_" + str(crop_size) + ".bin", MAP_FOLDER)
                if slide_size is None:
                    slide_size = dm.read_blob(basename + ".info", MAP_FOLDER)
                image = blnd.save_np_image(np_flatten_image, slide_size, crop_size)
                utils.save_image(image, SEGMENTED_FOLDER, basename + "_CS" + str(crop_size))

            prediction_images_path.append([path.join(SEGMENTED_FOLDER, filename + ".png"), path.join(SEGMENTED_FOLDER, filename_std_bi + ".png")])

            elapsed_time = time.time() - start_time
            log.print_debug(basename + "_CS" + str(crop_size) + "_DR" + str(dr).replace("0.", "") + "_ITER" + str(iterations) + "_gradient" + ": " + str(elapsed_time))
            gc.collect()
        else:
            log.print_debug(file + "doesn't exist.")
    return prediction_images_path

# Produce predictions samples provided in MAP folder
def produce_images():
    crop_sizes = [2240, 4480]
    iter = [1, 10]
    drop_rates = [0.01, 0.1, 0.5]
    blnd.print_multiple_gradient()
    for dr in drop_rates:
        for crop_size in crop_sizes:
            for i in range(1, 6):
                for iterations in iter:
                    start_time = time.time()
                    log.print_debug(
                        "PROCESSING map_" + str(i) + ".svs dr:" + str(dr) + " iter:" + str(iterations) + " Crop Size" + str(
                            crop_size))
                    make_prediction([path.join(MAP_FOLDER,"map_" + str(i) + ".svs")],crop_size,dr,iterations)
                    elapsed_time = time.time() - start_time
                    log.print_debug("map_" + str(i) + "_CS" + str(crop_size) + "_DR" + str(dr).replace("0.", "") + "_ITER" + str(iterations) + " processed" + ": " + str(elapsed_time))
                    gc.collect()


def main():
    utils.test_folder()

    return 0


if __name__ == "__main__":
   main()
