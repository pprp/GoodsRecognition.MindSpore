import os
import matplotlib.pyplot as plt
import numpy as np
import json


train_path = "/home/pdluser/dataset/all/train"
test_path = "/home/pdluser/dataset/all/test"

id2name_file = "/home/pdluser/dataset/all/id2name.json"

with open(id2name_file, "r") as f:
    id2name = json.load(f)


def process(path, save):
    a = dict()
    x = []
    y = []
    p = []
    for idx, item in enumerate(os.listdir(path)):
        class_path = os.path.join(path, item)
        num_files = len(os.listdir(class_path))
        x.append(idx)
        p.append(class_path)
        y.append(num_files)
        a[idx] = num_files

    plt.figure()
    plt.plot(x, y)
    # plt.imshow()

    print("max idx:", x[np.array(y).argmax()],
          " path:", p[np.array(y).argmax()])

    print(" name:", id2name['12'])

    plt.savefig(save)


process(train_path, 'train.png')
process(test_path, 'test.png')
