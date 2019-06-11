# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os
import sys
import utils
import filter

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
    s = slide.open_slide(DATASET_FOLDER+"/2_AC_1.svs")
    print("\n.", end="")
    image = slide.slide_to_image(s)
    print(".", end="")
    image_rgb = image.convert('RGB')
    print(".", end="")
    image_resized = slide.resize_image_r(image_rgb, slide.SCALE_FACTOR)
    print(".", end="")
    np_rgb = utils.image_to_np_rgb(image_resized)
    print(".", end="")
    np_grayscale = filter.filter_np_rgb_to_grayscale(np_rgb)
    np_complemented = filter.complement_np(np_grayscale)
    image_grayscale_complemented = utils.np_to_pil(np_complemented, utils.GRAY_SCALE)
    utils.save_image(image_grayscale_complemented, "filter", "temp.png")


if __name__ == "__main__":
    main()
