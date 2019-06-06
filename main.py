# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import slide
import os

DATASET_FOLDER = "dataset"


def main():
    print("Opening Whole-slide")

    # Production
    # for filename in os.listdir(DATASET_FOLDER):
    #     if filename.endswith(".svs"):
    #         slide.slide_info(DATASET_FOLDER + "/" + filename)

    # Debug
    slide.slide_info(DATASET_FOLDER + "/2_AC_1.svs")
    slide.crop(DATASET_FOLDER + "/2_AC_1.svs")


if __name__ == "__main__":
    main()
