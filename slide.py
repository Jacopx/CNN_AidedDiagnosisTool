# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide


def slide_info(slide_path):
    slide = openslide.open_slide(slide_path)
    width = int(slide.dimensions[0])
    height = int(slide.dimensions[1])
    print(slide.properties)
