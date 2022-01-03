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
ResNet based ResNext
"""
import math

import mindspore.nn as nn
from mindspore.common import initializer as init
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops.operations import Add, Concat, Split
from src.utils.cunstom_op import GlobalAvgPooling, GroupConv, SEBlock
from src.utils.var_init import KaimingNormal, default_recurisive_init

__all__ = ["ResNet", "se_resnext50"]


def weight_variable(shape, factor=0.1):
    return TruncatedNormal(0.02)


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        has_bias=has_bias,
        padding=padding,
        pad_mode="pad",
        group=groups,
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        has_bias=has_bias,
        padding=padding,
        pad_mode="pad",
        group=groups,
    )


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        has_bias=has_bias,
        padding=padding,
        pad_mode="pad",
        group=groups,
    )


class _DownSample(nn.Cell):
    """
    Downsample for ResNext-ResNet.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the 1*1 convolutional layer.

    Returns:
        Tensor, output tensor.

    Examples:
        >>>DownSample(32, 64, 2)
    """

    def __init__(self, in_channels, out_channels, stride):
        super(_DownSample, self).__init__()
        self.conv = conv1x1(in_channels, out_channels,
                            stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class BasicBlock(nn.Cell):
    """
    ResNet basic block definition.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>>BasicBlock(32, 256, stride=2)
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        down_sample=None,
        use_se=True,
        platform="Ascend",
        **kwargs
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = P.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)
        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True
        self.add = Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.down_sample_flag:
            identity = self.down_sample(x)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    """
    ResNet Bottleneck block definition.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the initial convolutional layer. Default: 1.

    Returns:
        Tensor, the ResNet unit's output.


        >>>Bottleneck(3, 256, stride=2)
    """

    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        down_sample=None,
        base_width=64,
        groups=1,
        use_se=True,
        platform="Ascend",
        **kwargs
    ):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.0)) * groups
        self.groups = groups
        self.conv1 = conv1x1(in_channels, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = P.ReLU()
        self.conv3x3s = nn.CellList()
        self.conv2 = GroupConv(width, width, 3, stride, pad=1, groups=groups)
        self.op_split = Split(axis=1, output_num=self.groups)
        self.op_concat = Concat(axis=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels * self.expansion)
        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True
        self.cast = P.Cast()
        self.add = Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)
        if self.down_sample_flag:
            identity = self.down_sample(x)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (cell): Block for network.
        layers (list): Numbers of block in different layers.
        width_per_group (int): Width of every group.
        groups (int): Groups number.

    Returns:
        Tuple, output tensor tuple.

    Examples:
        >>>ResNet()
    """

    def __init__(
        self,
        block,
        layers,
        num_classes,
        width_per_group=64,
        groups=1,
        use_se=True,
        is_drop=False,
        platform="Ascend",
        is_traing=True
    ):
        super(ResNet, self).__init__()
        self.is_traing = is_traing
        self.in_channels = 64
        self.groups = groups
        self.is_drop = is_drop
        self.base_width = width_per_group
        self.conv = conv7x7(3, self.in_channels, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(
            block, 64, layers[0], use_se=use_se, platform=platform
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, use_se=use_se, platform=platform
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, use_se=use_se, platform=platform
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, use_se=use_se, platform=platform
        )
        self.out_channels = 512 * block.expansion
        self.cast = P.Cast()
        self.gvp = GlobalAvgPooling()

        if self.is_drop:
            self.dropout = nn.Dropout(keep_prob=0.5)

        self.fc = nn.Dense(
            self.out_channels, num_classes, has_bias=False
        ).add_flags_recursive(fp16=True)
        default_recurisive_init(self)

    def init_weight(self):
        for cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(
                        KaimingNormal(
                            a=math.sqrt(5), mode="fan_out", nonlinearity="relu"
                        ),
                        cell.weight.shape,
                        cell.weight.dtype,
                    )
                )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape))
        # Zero-initialize the last BN in each residual branch,

        for cell in self.cells_and_names():
            if isinstance(cell, Bottleneck):
                cell.bn3.gamma.set_data(init.initializer(
                    "zeros", cell.bn3.gamma.shape))
            elif isinstance(cell, BasicBlock):
                cell.bn2.gamma.set_data(init.initializer(
                    "zeros", cell.bn2.gamma.shape))

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = self.gvp(x)
        if self.is_drop:
            feature = self.dropout(feature)
        x = self.fc(feature)
        if self.is_traing:
            return x
        else:
            return x, feature

    def _make_layer(
        self, block, out_channels, blocks_num, stride=1, use_se=True, platform="Ascend"
    ):
        """_make_layer"""
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = _DownSample(
                self.in_channels, out_channels * block.expansion, stride=stride
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                down_sample=down_sample,
                base_width=self.base_width,
                groups=self.groups,
                use_se=use_se,
                platform=platform,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks_num):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    base_width=self.base_width,
                    groups=self.groups,
                    use_se=use_se,
                    platform=platform,
                )
            )
        return nn.SequentialCell(layers)

    def get_out_channels(self):
        return self.out_channels


def se_resnext50(num_classes=2388,is_traning=True):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        width_per_group=4,
        groups=7,
        is_traning=is_traning
    )

if __name__ == "__main__":
    net = se_resnext50()
    print(net)
