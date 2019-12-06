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
import src.preparation.slide as slide
import src.preparation.utils as utils
import src.cnn.testCNN as convNet
from src.preparation import datasetManager as dm
from src.parameters import *

"""
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
        #grayscale -> gaussian
        for s in range(0, 11):
            #np_grayscale = filter.filter_np_rgb_to_grayscale(filter.gaussian_filter(np_rgb, s/10, 4))
            np_otsu_sk = filter.otsu_filter(np_normalized)
            np_otsu_sk_complemented = filter.complement_np(np_otsu_sk)
            utils.np_to_pil(filter.apply_mask(np_rgb, np_otsu_sk_complemented), utils.COLOR).show()

        #grayscale->otsu->complement
        np_otsu_sk = filter.otsu_filter_scikit(np_grayscale)
        np_otsu_sk_complemented = filter.complement_np(np_otsu_sk)
        utils.np_to_pil(np_otsu_sk_complemented, utils.GRAY_SCALE).show()
        #grayscale->complement->otsu
"""


def main():
    start_time = time.time()
    image, prediction_matrix, valid_bit_matrix = dm.get_prediction_matrix_multithread_test("map_2.svs")
    log.print_info('Prediction done')
    prediction_mask = utils.get_prediction_mask(prediction_matrix, valid_bit_matrix)
    prediction_mask_img = utils.np_to_pil(prediction_mask, utils.COLOR_ALPHA)
    print(prediction_mask_img.size)
    print(image.size)
    utils.save_image(prediction_mask_img, RESOURCE_FOLDER, "test_mask_2_2")
    utils.save_image(image, RESOURCE_FOLDER, "test_img_2_2")
    blended_image = utils.blend(image, prediction_mask_img)
    utils.save_image(utils.np_to_pil(blended_image, utils.COLOR_ALPHA), RESOURCE_FOLDER, "test_blend_2_2")

    """
    X_train , y_train, X_test, y_test = dm.open_dataset()
    #X_train, y_train, X_test, y_test = dm.open_dummy_dataset()
    log.print_info(" TRAIN STATs")
    log.print_info(" Train set shape : " + str(X_train.shape) +" "+ str(y_train.shape))
    log.print_info(" Train set type : " + str(X_train.dtype))
    dm.print_stats(y_train)
    log.print_info(" TEST STATs")
    log.print_info(" Test set shape : " + str(X_test.shape) +" "+ str(y_test.shape) )
    log.print_info(" Test set type : " + str(X_test.dtype))
    dm.print_stats(y_test)
    convNet.compile_model(X_train , y_train, X_test, y_test)"""
    elapsed_time = time.time() - start_time
    log.print_debug(" TOTAL TIME FOR PROCESSING: " + str(elapsed_time))
    return 0

if __name__ == "__main__":
    main()
