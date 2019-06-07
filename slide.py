# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide

SAMPLE_SIZE = 64


def slide_info(slide_path):
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])
    # print(slide.properties)
    w_sample = width // SAMPLE_SIZE + 1
    h_sample = height // SAMPLE_SIZE + 1
    print(slide_path, end=" ==> ")
    """print(width, height)
    print(w_sample, h_sample)"""
    return width, height
