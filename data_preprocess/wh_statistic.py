import math
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

w_dic_50 = defaultdict(int)
w_dic_500 = defaultdict(int)
h_dic_50 = defaultdict(int)
h_dic_100 = defaultdict(int)
h_dic_500 = defaultdict(int)
scale_dic = defaultdict(int)
scale_2_dic = defaultdict(int)
scale_set = set()

data_dir = "/home/pdluser/dataset/all/train"
for c_dir in os.listdir(data_dir):
    _path = os.path.join(data_dir, c_dir)
    if os.path.isdir(_path):
        for file in os.listdir(_path):
            file = os.path.join(_path, file)
            img = Image.open(file)
            w = img.width
            h = img.height
            scale = round(w / h, 2)
            scale_2 = round(h / w, 2)

            if w < 500:
                w_dic_50[w // 50] += 1
            elif w >= 500 and w < 1500:
                w_dic_500[w // 500] += 1

            if h < 700:
                h_dic_50[h // 50] += 1
            elif h >= 700 and h < 1000:
                h_dic_100[h // 100] += 1
            elif h >= 1000 and h < 2000:
                h_dic_500[h // 500] += 1

            scale_dic[scale // 0.5] += 1

            if scale_2 < 5:
                scale_2_dic[scale_2 // 0.5] += 1

w_num = max(w_dic_500.keys())
h_num = max(h_dic_500.keys())
scale_num = int(max(scale_dic))
scale_num_2 = int(max(scale_2_dic))


def draw(
    size,
    data_dic,
    title,
    xlabel,
    ylabel,
    save_name,
    rotation=0,
    figsize=(10, 5),
    label_base=500,
    base_data={},
    base_num=0,
    base_500_1000={},
    base_num_2=0,
):
    data_lis = []
    labels = []

    if base_data:
        for i in range(base_num):
            data_lis.append(base_data[i])
            labels.append("%s-%s" % (i * 50, (i + 1) * 50))

    if base_500_1000:
        for i in range(7, 10):
            data_lis.append(base_500_1000[i])
            labels.append("%s-%s" % (i * 100, (i + 1) * 100))

    for i in range(size + 1):
        l = []
        if base_num:
            l = [0]
        if base_num and base_num_2:
            l = [0, 1]
        if base_data and i in l:
            continue

        data_lis.append(data_dic[i])
        labels.append("%s-%s" % (i * label_base, (i + 1) * label_base))

    fig = plt.figure(figsize=figsize)

    plt.bar(
        range(len(data_lis)),
        data_lis,
        width=0.5,
        color=["coral", "dodgerblue", "gray", "thistle", "palegreen"],
        tick_label=labels,
    )

    for a, b in zip(range(size + 1 + base_num + base_num_2), data_lis):  # 柱子上的数字显示
        plt.text(a, b, "%d" % b, ha="center", va="bottom", fontsize=7)

    plt.xticks(rotation=rotation)  # lable旋转

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_name)
    plt.clf()  # 添加上这一行，画完第一个图后，将plt重置


draw(
    w_num,
    w_dic_500,
    "width distribution diagram",
    "width",
    "number",
    "w_bar.png",
    figsize=(20, 5),
    base_data=w_dic_50,
    base_num=10,
)
draw(
    h_num,
    h_dic_500,
    "height distribution diagram",
    "height",
    "number",
    "h_bar.png",
    figsize=(20, 5),
    base_data=h_dic_50,
    base_num=14,
    base_500_1000=h_dic_100,
    base_num_2=3,
)
draw(
    scale_num,
    scale_dic,
    "Distribution diagram of aspect ratio",
    "w/h",
    "number",
    "scale_bar.png",
    label_base=0.5,
)
draw(
    scale_num_2,
    scale_2_dic,
    "Distribution diagram of aspect ratio",
    "h/w",
    "number",
    "scale_bar_2.png",
    label_base=0.5,
    figsize=(20, 5),
)
plt.show()
