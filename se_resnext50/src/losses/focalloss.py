import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn.loss import LossBase
import mindspore

# focal loss: afa=2, beta=4
class FocalLoss(LossBase):
    """nn.Cell warpper for focal loss"""

    def __init__(self, alpha=2.0, beta=4.0):
        super(FocalLoss, self).__init__()
        self.log = P.Log()
        self.pow = P.Pow()
        self.sum = P.ReduceSum()
        self.shape = P.Shape()
        self.afa = alpha
        self.beta = beta

    def construct(self, pred, gt):
        # (bs, classes) bs
        print(pred.shape, gt.shape)  # 128,1864 ; 128

        """Construct method"""
        pos_inds = P.Select()(
            P.Equal()(gt, 1.0),
            P.Fill()(P.DType()(gt), P.Shape()(gt), 1.0),
            P.Fill()(P.DType()(gt), P.Shape()(gt), 0.0),
        )
        neg_inds = P.Select()(
            P.Less()(gt, 1.0),
            P.Fill()(P.DType()(gt), P.Shape()(gt), 1.0),
            P.Fill()(P.DType()(gt), P.Shape()(gt), 0.0),
        )
        print(pos_inds.shape, neg_inds.shape)  # 128, 128

        neg_weights = self.pow(1 - gt, self.beta)  # beta=4
        # afa=2

        print(self.log(pred).shape)
        print(self.pow(1 - pred, self.afa).shape)
        print(pos_inds.shape)

        pos_loss = self.log(pred) * self.pow(1 - pred, self.afa)
        pos_loss = pos_loss * pos_inds
        neg_loss = (
            self.log(1 - pred) * self.pow(pred, self.afa) * neg_weights * neg_inds
        )

        num_pos = self.sum(pos_inds, ())
        num_pos = P.Select()(
            P.Equal()(num_pos, 0.0),
            P.Fill()(P.DType()(num_pos), P.Shape()(num_pos), 1.0),
            num_pos,
        )

        pos_loss = self.sum(pos_loss, ())
        neg_loss = self.sum(neg_loss, ())
        loss = -(pos_loss + neg_loss) / num_pos
        return loss


class CustomNetWithLossCell(nn.Cell):
    """
    Centerface with loss cell.
    """

    def __init__(self, config, network):
        super(CustomNetWithLossCell, self).__init__()
        self.network = network
        self.config = config
        self.loss = FocalLoss2(self.config.afa, self.config.beta)
        self.reduce_sum = P.ReduceSum()

    def construct(self, x, label):
        print(x.shape, label.shape)
        output = self.network(x)
        loss = self.loss(output, label)
        return loss


class FocalLoss2(LossBase):
    """
    Warpper for focal loss.
    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.
    Returns:
        Tensor, focal loss.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(FocalLoss2, self).__init__()
        self.alpha = self.cast(alpha, mstype.float32)
        self.beta = self.cast(beta, mstype.float32)
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()
        self.broadcast = ops.Broadcast(1)
        self.reshape = ops.Reshape()

    def construct(self, out, target):
        """focal loss"""
        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        neg_inds = self.cast(self.less(target, 1.0), mstype.float32)

        neg_weights = self.pow(1 - target, self.beta)

        print(self.log(out).shape)
        print(self.pow(1 - out, self.alpha).shape)
        print(pos_inds.shape)



        neg_loss = (
            self.log(1 - out) * self.pow(out, self.alpha) * neg_weights * neg_inds
        )

        pos_loss = self.log(out) * self.pow(1 - out, self.alpha) 

        # c = self.shape(pos_inds)[0]
        # pos_inds = self.reshape(pos_inds, (c , 1864))
        # print(pos_inds.shape)
        pos_loss = pos_loss * pos_inds

        num_pos = self.reduce_sum(pos_inds, ())
        num_pos = self.select(
            self.equal(num_pos, 0.0),
            self.fill(self.dtype(num_pos), self.shape(num_pos), 1.0),
            num_pos,
        )
        pos_loss = self.reduce_sum(pos_loss, ())
        neg_loss = self.reduce_sum(neg_loss, ())
        loss = -(pos_loss + neg_loss) / num_pos
        return loss


class FocalLoss3(LossBase):
    """
    Warpper for focal loss.
    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.
    Returns:
        Tensor, focal loss.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(FocalLoss2, self).__init__()
        self.alpha = self.cast(alpha, mstype.float32)
        self.beta = self.cast(beta, mstype.float32)
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()
        self.broadcast = ops.Broadcast(1)
        self.reshape = ops.Reshape()
        self.gather = ops.GatherD()
        self.exp = ops.Exp()

    def construct(self, input, target):
        """focal loss"""
        target = target.view(-1, 1)

        logpt = nn.LogSoftmax(input)
        logpt = self.gather(logpt, 1, target)
        logpt = logpt.view(-1)
        pt = self.exp(logpt)#Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

class FocalLoss4(LossBase):

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss4, self).__init__(reduction=reduction)
        # 校验gamma，这里的γ称作focusing parameter，γ>=0，称为调制系数
        self.gamma = gamma #validator.check_value_type("gamma", gamma, [float])
        if weight is not None and not isinstance(weight, Tensor):
            raise TypeError("The type of weight should be Tensor, but got {}.".format(type(weight)))
        self.weight = weight
        # 用到的mindspore算子
        self.expand_dims = P.ExpandDims()
        self.gather_d = P.GatherD()
        self.squeeze = P.Squeeze(axis=1)
        self.tile = P.Tile()
        self.cast = P.Cast()

    def construct(self, predict, target):
        targets = target
        # 对输入进行校验
        # _check_ndim(predict.ndim, targets.ndim)
        # _check_channel_and_shape(targets.shape[1], predict.shape[1])
        # _check_predict_channel(predict.shape[1])

        # 将logits和target的形状更改为num_batch * num_class * num_voxels.
        if predict.ndim > 2:
            predict = predict.view(predict.shape[0], predict.shape[1], -1) # N,C,H,W => N,C,H*W
            targets = targets.view(targets.shape[0], targets.shape[1], -1) # N,1,H,W => N,1,H*W or N,C,H*W
        else:
            predict = self.expand_dims(predict, 2) # N,C => N,C,1
            targets = self.expand_dims(targets, 2) # N,1 => N,1,1 or N,C,1
        
        # 计算对数概率
        log_probability = nn.LogSoftmax(1)(predict)
        # 只保留每个voxel的地面真值类的对数概率值。
        if target.shape[1] == 1:
            log_probability = self.gather_d(log_probability, 1, self.cast(targets, mindspore.int32))
            log_probability = self.squeeze(log_probability)

        # 得到概率
        probability = F.exp(log_probability)

        if self.weight is not None:
            convert_weight = self.weight[None, :, None]  # C => 1,C,1
            convert_weight = self.tile(convert_weight, (targets.shape[0], 1, targets.shape[2])) # 1,C,1 => N,C,H*W
            if target.shape[1] == 1:
                convert_weight = self.gather_d(convert_weight, 1, self.cast(targets, mindspore.int32))  # selection of the weights  => N,1,H*W
                convert_weight = self.squeeze(convert_weight)  # N,1,H*W => N,H*W
            # 将对数概率乘以它们的权重
            probability = log_probability * convert_weight
        # 计算损失小批量
        weight = F.pows(-probability + 1.0, self.gamma)
        if target.shape[1] == 1:
            loss = (-weight * log_probability).mean(axis=1)  # N
        else:
            loss = (-weight * targets * log_probability).mean(axis=-1)  # N,C

        return self.get_loss(loss)