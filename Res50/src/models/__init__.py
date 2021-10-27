from .resnet import resnet50, resnet101
from .GENet import GE_resnet50

_model_factory = {
    'resnet50': resnet50, 
    'resnet101': resnet101,
    'geresnet50': GE_resnet50
}

def build_network(name, num_class=2390, width=32):
    return _model_factory[name](num_class, width)

