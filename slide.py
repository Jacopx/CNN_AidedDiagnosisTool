# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide
import time
import os

SS = 256  # Sample Size
LVL = 0   # Level of dimensionality
CROP_FOLDER = 'crop'


def slide_info(slide_path):
    print("Whole-slide info:")
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])
    # print(slide.properties)
    # print(slide.level_dimensions)

    print(slide_path, end=" ==> ")
    print(width, height, end=" / ")

    # Computing overlapping for width
    w_sample = width / SS
    dec_w = w_sample % 1
    over_w = dec_w * SS

    # Computing overlapping for height
    h_sample = height / SS
    dec_h = h_sample % 1
    over_h = dec_h * SS

    print(str(w_sample) + "::" + str(h_sample), end=" || ")
    print(over_w, over_h, end="\n\n")


def crop(folder, file_folder, slide):
    start_time = time.time()
    slide_path = folder + "/" + slide
    try:
        # Create target Directory
        os.mkdir(CROP_FOLDER + "/" + file_folder)
        print("Directory ", CROP_FOLDER + "/" + file_folder, " Created ")
    except FileExistsError:
        print("Directory ", CROP_FOLDER + "/" + file_folder, " already exists")

    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])

    # Computing overlapping for width
    w_sample = width / SS
    dec_w = w_sample % 1
    over_w = dec_w * SS

    # Computing overlapping for height
    h_sample = height / SS
    dec_h = h_sample % 1
    over_h = dec_h * SS

    print("Get image...")
    image = slide.read_region((0, 0), LVL, (width, height))
    print("Resize image...")
    image.resize((int(width-over_w), int(height-over_h)))

    print("Starting crop...")
    # # DEBUG CYCLE
    # for shift_h in range(0, 10):
    #     for shift_w in range(0, 10):
    crop_number = 0
    for shift_w in range(0, int(w_sample)):
        for shift_h in range(0, int(h_sample)):
            if shift_h == 0 and shift_w == 0:
                box = (0, 0, SS, SS)
            elif shift_h == 0:
                box = (SS * shift_w, 0, SS * (shift_w + 1), SS)
            elif shift_h == 0:
                box = (0, SS * shift_h, 0, SS * (shift_h + 1))
            else:
                box = (SS * shift_w, SS * shift_h, SS * (shift_w + 1), SS * (shift_h + 1))

            print(box, end=" ==> ")
            crop_region = image.crop(box)

            crop_name = CROP_FOLDER + "/" + file_folder + '/crop_' + str(shift_w) + 'x' + str(shift_h) + '.png'
            crop_region.save(crop_name)
            print(str(shift_w) + 'x' + str(shift_h))
            crop_number += 1

    elapsed_time = time.time() - start_time
    print("\n # Crop: " + str(crop_number) + " || Time Elapsed: " + str(elapsed_time))
