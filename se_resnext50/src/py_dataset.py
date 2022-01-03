# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dataset processing.
"""
import os

import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.common import dtype as mstype
from mindspore.dataset.vision.py_transforms import Cutout, RandomErasing
from PIL import Image, ImageFile

from src.utils.augment import _pil_interp, rand_augment_transform
from src.utils.sampler import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TxtDataset:
    """
    create txt dataset.

    Args:
    Returns:
        de_dataset.
    """

    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        fin = open(txt_name, "r")
        for line in fin:
            img_name, label = line.strip().split(" ")
            self.imgs.append(os.path.join(root, img_name))
            self.labels.append(int(label))
        fin.close()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)


def classification_dataset(
        data_dir,
        image_size,
        per_batch_size,
        max_epoch,
        rank,
        group_size,
        mode="train",
        input_mode="folder",
        root="",
        num_parallel_workers=4,
        shuffle=None,
        sampler=None,
        class_indexing=None,
        drop_remainder=True,
        transform=None,
        target_transform=None,
        config=None,
):
    """
    A function that returns a dataset for classification. The mode of input dataset could be "folder" or "txt".
    If it is "folder", all images within one folder have the same label. If it is "txt", all paths of images
    are written into a textfile.

    Args:
        data_dir (str): Path to the root directory that contains the dataset for "input_mode="folder"".
            Or path of the textfile that contains every image's path of the dataset.
        image_size (Union(int, sequence)): Size of the input images.
        per_batch_size (int): the batch size of evey step during training.
        max_epoch (int): the number of epochs.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided
            into (default=None).
        mode (str): "train" or others. Default: " train".
        input_mode (str): The form of the input dataset. "folder" or "txt". Default: "folder".
        root (str): the images path for "input_mode="txt"". Default: " ".
        num_parallel_workers (int): Number of workers to read the data. Default: None.
        shuffle (bool): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        sampler (Sampler): Object used to choose samples from the dataset. Default: None.
        class_indexing (dict): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).

    Examples:
        >>> from src.dataset import classification_dataset
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> data_dir = "/path/to/imagefolder_directory"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4)
        >>> # Path of the textfile that contains every image's path of the dataset.
        >>> data_dir = "/path/to/dataset/images/train.txt"
        >>> images_dir = "/path/to/dataset/images"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4,
        >>>                               input_mode="txt", root=images_dir)
    """
    mean = [0.4716407, 0.42498824, 0.35398316]
    std = [0.21783721, 0.2078618, 0.19345342]
    if transform is None:
        if mode == "train":
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                py_vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)
                ),
                py_vision.RandomHorizontalFlip(prob=0.5),
                py_vision.RandomColorAdjust(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
            ]

            if config:
                if config.auto_augment:
                    aa_params = dict(
                        translate_const=int(image_size[0] * 0.45),
                        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
                    )
                    interpolation = config.interpolation
                    auto_augment_config = config.auto_augment_config
                    assert auto_augment_config.startswith('rand')
                    aa_params['interpolation'] = _pil_interp(interpolation)
                    transform_img.append(rand_augment_transform(auto_augment_config, aa_params))

            transform_img += [
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
            ]

            if config:
                if config.random_erase:
                    transform_img.append(RandomErasing())

                if config.cutout:
                    transform_img.append(Cutout(config.cutout_length))
        else:
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                # py_vision.Resize((256, 256)),
                py_vision.CenterCrop(image_size),
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
                # py_vision.HWC2CHW(),
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    if input_mode == "folder":
        de_dataset = de.ImageFolderDataset(
            data_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
            sampler=sampler,
            class_indexing=class_indexing,
            num_shards=group_size,
            shard_id=rank,
        )
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(
        operations=transform_img,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
    )
    de_dataset = de_dataset.map(
        operations=transform_label,
        input_columns="label",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
    )

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset


class Resize_with_Ratio(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
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

def TTA_classification_dataset(
        data_dir,
        image_size,
        per_batch_size,
        max_epoch,
        rank,
        group_size,
        mode="eval1", # choice eval1 eval2 ..
        input_mode="folder",
        root="",
        num_parallel_workers=4,
        shuffle=None,
        sampler=None,
        class_indexing=None,
        drop_remainder=True,
        transform=None,
        target_transform=None,
        config=None,
):
   
    mean = [0.4716407, 0.42498824, 0.35398316]
    std = [0.21783721, 0.2078618, 0.19345342]
    if transform is None:
        if mode == "eval1":
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                py_vision.RandomColorAdjust(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                py_vision.Resize((256, 256)),
                py_vision.CenterCrop(image_size),
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
            ]
        elif mode == "eval2":
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                py_vision.RandomHorizontalFlip(prob=1),
                py_vision.CenterCrop(image_size),
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
            ] 
        else:
            raise "mode should be eval1 or eval2"
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    if input_mode == "folder":
        de_dataset = de.ImageFolderDataset(
            data_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
            sampler=sampler,
            class_indexing=class_indexing,
            num_shards=group_size,
            shard_id=rank,
        )
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(
        operations=transform_img,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
    )
    de_dataset = de_dataset.map(
        operations=transform_label,
        input_columns="label",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
    )

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset

def arcface_classification_dataset(
        data_dir,
        image_size,
        per_batch_size,
        max_epoch,
        rank,
        group_size,
        mode="train",
        input_mode="folder",
        root="",
        num_parallel_workers=4,
        shuffle=None,
        sampler=None,
        class_indexing=None,
        drop_remainder=True,
        transform=None,
        target_transform=None,
        config=None,
):
    mean = [0.4716407, 0.42498824, 0.35398316]
    std = [0.21783721, 0.2078618, 0.19345342]
    if transform is None:
        if mode == "train":
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                py_vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)
                ),
                py_vision.RandomHorizontalFlip(prob=0.5),
                py_vision.RandomColorAdjust(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
            ]

            if config:
                if config.auto_augment:
                    aa_params = dict(
                        translate_const=int(image_size[0] * 0.45),
                        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
                    )
                    interpolation = config.interpolation
                    auto_augment_config = config.auto_augment_config
                    assert auto_augment_config.startswith('rand')
                    aa_params['interpolation'] = _pil_interp(interpolation)
                    transform_img.append(rand_augment_transform(auto_augment_config, aa_params))

            transform_img += [
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
            ]

            if config:
                if config.random_erase:
                    transform_img.append(RandomErasing())

                if config.cutout:
                    transform_img.append(Cutout(config.cutout_length))
        else:
            transform_img = [
                py_vision.Decode(),
                Resize_with_Ratio((256, 256)),
                # py_vision.Resize((256, 256)),
                py_vision.CenterCrop(image_size),
                py_vision.ToTensor(),
                py_vision.Normalize(mean=mean, std=std),
                # py_vision.HWC2CHW(),
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    if input_mode == "folder":
        de_dataset = de.ImageFolderDataset(
            data_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
            sampler=sampler,
            class_indexing=class_indexing,
            num_shards=group_size,
            shard_id=rank,
        )
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(
        operations=transform_img,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
        # python_multiprocessing=True,
    )
    de_dataset = de_dataset.map(
        operations=transform_label,
        input_columns="label",
        num_parallel_workers=num_parallel_workers,
        # python_multiprocessing=True,
    )

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset