from .focalloss import FocalLoss, FocalLoss2, FocalLoss4
from .crossentropy import CrossEntropy
from .labelsmooth import CrossEntropySmooth
import mindspore.nn as nn 
from mindspore import Tensor
import numpy as np
from .arcface import ArcFaceLoss

_loss_factory = {
    'ce': CrossEntropy,
    'ls': CrossEntropySmooth,
    'fl': FocalLoss4,
    'af': ArcFaceLoss,
}

def build_loss(config):
    assert config.loss_type in ['ce', 'ls', 'fl', 'af']
    if config.loss_type == 'ce':
        loss = _loss_factory[config.loss_type]()
    elif config.loss_type == 'ls':
        loss = _loss_factory[config.loss_type](smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    elif config.loss_type == 'fl':
        loss = _loss_factory[config.loss_type](weight=Tensor(np.ones([128,])), gamma=config.beta)
    elif config.loss_type == 'af':
        loss = _loss_factory[config.loss_type](world_size=1,s=30.0, m=0.5)
    else:
        raise "NOT IMPLEMENTED"
    return loss 