# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import openslide
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

DATASET_FOLDER = sys.argv[1]

##TODO : CLEAN CODE!!!!


def image_size_plot(folder):
    # Function use to plot dimensions over a scatter plot of the image size for the test set

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


def print_cropped_stats():
    h_b, ad_b, ac_b = distributions(DATASET_FOLDER, ".svs")
    h_o, ad_o, ac_o = distributions("cropped_dataset/overlap_no_padding/3190", ".png")
    h_r, ad_r, ac_r = distributions("cropped_dataset/resize_crop/3190", ".png")

    t_b = h_b + ad_b + ac_b
    t_o = h_o + ad_o + ac_o
    t_r = h_r + ad_r + ac_r

    b = (round(ad_b/t_b*100 + ac_b/t_b*100, 2), round(h_b/t_b*100, 2), round(ad_b/t_b*100, 2), round(ac_b/t_b*100, 2))
    o = (round(ad_o/t_o*100 + ac_o/t_o*100, 2), round(h_o/t_o*100, 2), round(ad_o/t_o*100, 2), round(ac_o/t_o*100, 2))
    r = (round(ad_r/t_r*100 + ac_r/t_r*100, 2), round(h_r/t_r*100, 2), round(ad_r/t_r*100, 2), round(ac_r/t_r*100, 2))

    ind = np.arange(len(b))  # the x locations for the groups
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width, b, width, label='Base Dataset', color='#003B46')
    rects2 = ax.bar(ind, o, width, label='Overlapped crop', color='#07575B')
    rects3 = ax.bar(ind + width, r, width, label='Resized crop', color='#66A5AD')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage over total')
    ax.set_title('Classes partition by crop algorithm and size')
    ax.set_xticks(ind)
    ax.set_xticklabels(('NH','H', 'AD', 'AC'))
    ax.legend()
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha='center', va='bottom')
    for rect in rects3:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.show()


def distributions(folder, image_type):
    h_c = 0
    ad_c = 0
    ac_c = 0
    for filename in os.listdir(folder):
        if filename.endswith(image_type):
            if "_H_" in filename:
                h_c += 1
            elif "_AC_" in filename:
                ac_c += 1
            elif "_AD_" in filename:
                ad_c += 1
    total = h_c + ad_c + ac_c
    print("-------------------------------------------------------------------")
    print("Total of " + folder + ": " + str(total))
    print("Health: " + str(h_c) + "(" + str(round(h_c/total*100, 2)) + "%)")
    print("Adenoma: " + str(ad_c) + "(" + str(round(ad_c/total*100, 2)) + "%)")
    print("Adenocarcinoma: " + str(ac_c) + "(" + str(round(ac_c/total*100, 2)) + "%)")
    return h_c, ad_c, ac_c


if __name__ == "__main__":
    image_size_plot()
