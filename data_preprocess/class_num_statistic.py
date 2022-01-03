import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import numpy as np 
from matplotlib.pyplot import MultipleLocator


def statistic_dir(data_dir):
    lis = []
    for c_dir in os.listdir(data_dir):
        _path = os.path.join(data_dir, c_dir)
        if os.path.isdir(_path):
            n = len(os.listdir(_path))
            lis.append(n)
    return lis


def plot_class_num(lis, fromidx=0, toidx=100):
    img_num = range(1, len(lis) + 1)
    sorted_list = sorted(lis)
    img_num = img_num[fromidx:toidx]
    sorted_list = sorted_list[fromidx:toidx]

    plt.figure(figsize=(20, 5), dpi=500)
    plt.bar(
        img_num,
        sorted_list,
        align="center",
        color="c",
        alpha=0.8,
        width=0.35,
        label="Image-Number",
    )

    for a, b in zip(img_num, sorted_list):
        plt.text(a, b + 0.05, "%.0f" % b, ha="center", va="bottom", fontsize=6)

    # plt.scatter(img_num, sorted(lis), color="red", label="number")
    ax = plt.gca()
    x_major_locator = MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.title("Distribution diagram of each class of data")
    # plt.xlabel("class")
    plt.xlim()
    plt.ylabel("number")
    plt.legend()
    plt.savefig("testclass_number_f%d_t%d.png" % (fromidx, toidx))


def plot_range_class_num(lis, fromidx, toidx):
    img_num = range(1, len(lis) + 1)
    sorted_list = np.array(sorted(lis))
    range_list = range(0, 3800, 100)

    X = []
    Y = []

    print("sum of number below 10:", sum(sorted_list[sorted_list < 10]))
    print("total num", sum(sorted_list))

    for i in range(1, len(range_list)):
        s = range_list[i-1]
        e = range_list[i]
        n = len(sorted_list[(sorted_list >= s) & (sorted_list < e)])
        X.append(e)
        Y.append(n)

    X = X[fromidx:toidx]
    Y = Y[fromidx:toidx]

    plt.bar(
        X,
        Y,
        align="center",
        color="c",
        alpha=0.8,
        width=0.35,
        label="Image-Number",
    )

    for a, b in zip(X, Y):
        plt.text(a, b + 0.05, "%.0f" % b, ha="center", va="bottom", fontsize=6)

    plt.scatter(X, Y, color="red", label="number")
    ax = plt.gca()
    x_major_locator = MultipleLocator(100)
    ax.xaxis.set_major_locator(x_major_locator)

    # plt.title("Distribution diagram of each class of data")
    # # plt.xlabel("class")
    plt.ylabel("number")
    plt.legend()
    plt.savefig("range num of classes")


def main():
    data_dir = "/home/pdluser/datasets/all/train"
    lst = statistic_dir(data_dir)

    # plot_range_class_num(lst, 0, 10)
    plot_class_num(lst,0,2000)


if __name__ == "__main__":
    main()
