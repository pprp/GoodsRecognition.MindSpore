from src.py_dataset import classification_dataset
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision import Inter
from PIL import Image


class Resize_with_Ratio(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # print("type:", type(img))
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)
        return img


def cal_mean_std(data_path):
    image_size = [224, 224]
    per_batch_size = 128
    rank = 0
    group_size = 1
    de_dataset = classification_dataset(
        data_path,
        image_size,
        per_batch_size,
        1,
        rank,
        group_size,
        num_parallel_workers=8,
        # transform=[
        #     c_vision.Decode(),
        #     c_vision.Resize(image_size, Inter.BICUBIC),
        #     py_vision.ToTensor(),
        # ]
        transform=[
            py_vision.Decode(),
            Resize_with_Ratio((256, 256)),
            py_vision.RandomResizedCrop(
                image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)
            ),
            py_vision.RandomHorizontalFlip(prob=0.5),
            py_vision.RandomColorAdjust(
                brightness=0.4, contrast=0.4, saturation=0.4
            ),
            py_vision.ToTensor(),
        ]
    )

    mean = 0.
    std = 0.
    n_sample = 0

    for _data, label in de_dataset:
        data = _data.asnumpy()
        # print(data.shape)
        bs = data.shape[0]
        data = data.reshape((bs, data.shape[1], -1))
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_sample += bs

    mean /= n_sample
    std /= n_sample

    print("mean: {}\nstd: {}".format(mean, std))


train_data = "/home/niu/data/all/train"
# eval_data = "/home/niu/data/all/test"
cal_mean_std(train_data)
# cal_mean_std(eval_data)
