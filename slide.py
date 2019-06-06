# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide

SS = 256  # Sample Size
LVL = 0   # Level of dimensionality
CROP_FOLDER = 'crop'


def slide_info(slide_path):
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
    print(over_w, over_h)


def crop(slide_path):
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])

    # Computing overlapping for width
    w_sample = width / SS
    # dec_w = w_sample % 1
    # over_w = dec_w * SS  #NOT GOOD

    # Computing overlapping for height
    h_sample = height / SS
    # dec_h = h_sample % 1
    # over_h = dec_h * SS  #NOT GOOD

    # The first cycle is for production, the second is for debug

    # for shift_w in range(0, int(w_sample)):
    #     for shift_h in range(0, int(h_sample)):
    for shift_w in range(0, 10):
        for shift_h in range(0, 10):
            # @TODO: Managing overlapping decisions
            position = (SS * shift_w, SS * shift_h)
            print(position, end="\t=\t")
            crop_region = slide.read_region(position, LVL, (SS, SS))

            crop_name = CROP_FOLDER + '/crop_' + str(shift_w) + 'x' + str(shift_h) + '.png'
            crop_region.save(crop_name)
            print('crop_' + str(shift_w) + 'x' + str(shift_h))

    print("\n # Crop: " + str(int(w_sample*h_sample)))
