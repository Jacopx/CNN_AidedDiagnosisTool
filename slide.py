# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide

SAMPLE_SIZE = 256
CROP_FOLDER = 'crop'


def slide_info(slide_path):
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])
    # print(slide.properties)

    # Computing overlapping for width
    w_sample = width / SAMPLE_SIZE
    dec_w = w_sample % 1
    over_w = dec_w * SAMPLE_SIZE

    # Computing overlapping for height
    h_sample = height / SAMPLE_SIZE
    dec_h = h_sample % 1
    over_h = dec_h * SAMPLE_SIZE

    print(slide_path, end=" ==> ")
    print(width, height, end=" / ")
    print(w_sample, h_sample, end=" :: ")
    print(over_w, over_h)

    # The first cycle is for production, the second is for debug
    # for shift_w in range(0, int(w_sample)):
    #     for shift_h in range(0, int(h_sample)):
    for shift_w in range(0, 10):
        for shift_h in range(0, 10):
            crop = slide.read_region((SAMPLE_SIZE*shift_w, SAMPLE_SIZE*shift_h), 2, (SAMPLE_SIZE, SAMPLE_SIZE))
            crop_name = CROP_FOLDER + '/crop_' + str(shift_w) + 'x' + str(shift_h) + '.png'
            crop.save(crop_name)
            print("Generating: " + 'crop_' + str(shift_w) + 'x' + str(shift_h))

    print("\n # Crop: " + str(int(w_sample*h_sample)))
