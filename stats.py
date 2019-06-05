# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide
import os
import matplotlib.pyplot as plt


def image_size_plot():
    # Function use to plot dimensions over a scatter plot of the image size for the test set
    folder = 'dataset'

    xH, yH = [], []
    xAC, yAC = [], []
    xAD, yAD = [], []

    for filename in os.listdir(folder):
        if filename.endswith(".svs"):
            if "_H_" in filename:
                slide = openslide.open_slide(folder + "/" + filename)
                xH.append(int(slide.dimensions[0]))
                yH.append(int(slide.dimensions[1]))
            elif "_AC_" in filename:
                slide = openslide.open_slide(folder + "/" + filename)
                xAC.append(int(slide.dimensions[0]))
                yAC.append(int(slide.dimensions[1]))
            elif "_AD_" in filename:
                slide = openslide.open_slide(folder + "/" + filename)
                xAD.append(int(slide.dimensions[0]))
                yAD.append(int(slide.dimensions[1]))

    plt.scatter(xH, yH, c='y', marker="p")
    plt.scatter(xAC, yAC, c='c', marker="v")
    plt.scatter(xAD, yAD, c='m', marker="x")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    image_size_plot()
