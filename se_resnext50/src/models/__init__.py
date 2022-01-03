from .effnet import (
    efficientnetb0,
    efficientnetb3,
    efficientnetb4,
    efficientnetb5,
    efficientnetb6,
    efficientnetb7,
)
from .resnet import resnet50, resnet101
from .resnet_bam import resnet50_bam, resnet50_cbam
from .resnext import resnext50, resnext101
from .se_resnet import se_resnet50
from .se_resnext import se_resnext50
from .wideresnet import wideresnet_d4w10
from .se_resnext_wider import se_resnext50_wider
from .incept4 import inceptionv4
from .incept_resnet_v2 import inception_resnet_v2
from .resnet_bam_wider import resnet50_bam_wider
from .resnet_bam_arcface import resnet50_bam_arcface
from .swintransformer import swin_tiny_patch4_window7_224
from .incept_resnet_v2_wider import inception_resnet_v2_wider

_model_factory = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "se_resnet50": se_resnet50,
    "se_resnext50": se_resnext50,
    "resnext50": resnext50,
    "resnext101": resnext101,
    "efficientnetb0": efficientnetb0,
    "efficientnetb7": efficientnetb7,
    "efficientnetb6": efficientnetb6,
    "efficientnetb5": efficientnetb5,
    "efficientnetb4": efficientnetb4,
    "efficientnetb3": efficientnetb3,
    "resnet50_bam": resnet50_bam,
    "resnet50_cbam": resnet50_cbam,
    "wideresnet_d4w10": wideresnet_d4w10,
    "se_resnext50_wider": se_resnext50_wider,
    "inceptionv4": inceptionv4,
    "inception_resnet_v2": inception_resnet_v2,
    "resnet50_bam_wider": resnet50_bam_wider,
    "resnet50_bam_arcface": resnet50_bam_arcface,
    "swintransformer": swin_tiny_patch4_window7_224,
    "inception_resnet_v2_wider": inception_resnet_v2_wider,
}


def build_network(name, num_classes=2388):
    if name not in _model_factory.keys():
        raise ValueError(
            "%s Not Implemented. Support %s" % (name, _model_factory.keys())
        )
    return _model_factory[name](num_classes)

    # # define network
    # net = EfficientNet(1, 1)
    # net.to_float(mstype.float16)
