# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os
import sys

DATASET_FOLDER = sys.argv[1]


def min_max_ss():
    size_list = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            w_temp, h_temp = slide.get_slidepath_size(DATASET_FOLDER + "/" + filename)
            size_list.append(w_temp)
            size_list.append(h_temp)
    return min(size_list)


def main():
    # Production

    ss = min_max_ss()
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            slide.overlap_crop_multithread(DATASET_FOLDER, filename, ss)

    
if __name__ == "__main__":
    main()
