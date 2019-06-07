# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os
import sys
import math
import functools as ft
import stats

DATASET_FOLDER = sys.argv[1]
CROP_SIZE = 250


def main():
    print("Opening Whole-slide dataset "+DATASET_FOLDER)
    w_list = []
    h_list = []
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".svs"):
            w_temp, h_temp = slide.slide_info(DATASET_FOLDER + "/" + filename)
            print(w_temp, h_temp, end=" ")
            t_r = w_temp % CROP_SIZE
            if t_r != 0:
                w_temp += CROP_SIZE - (w_temp % CROP_SIZE)
            print(t_r, end=" ")
            t_r = h_temp % CROP_SIZE
            if t_r != 0:
                h_temp += CROP_SIZE - (h_temp % CROP_SIZE)
            print(t_r, end=" ")
            print(w_temp, h_temp)

            w_list.append(w_temp)
            h_list.append(h_temp)

    print("W max: "+str(max(w_list)))
    print("W min: "+str(min(w_list)))
    print("H max: "+str(max(h_list)))
    print("H min: "+str(min(h_list)))
    w_gcd = ft.reduce(lambda x, y: math.gcd(x, y), w_list)
    h_gcd = ft.reduce(lambda x, y: math.gcd(x, y), h_list)
    print("W_GCD: "+str(w_gcd))
    print("H_GCD: "+str(h_gcd))
    print("GCD: "+str(math.gcd(w_gcd, h_gcd)))
    stats.image_size_plot(DATASET_FOLDER)


if __name__ == "__main__":
    main()
