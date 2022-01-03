import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

# focal loss: afa=2, beta=4
class FocalLoss(nn.Cell):
    '''nn.Cell warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.log = P.Log()
        self.pow = P.Pow()
        self.sum = P.ReduceSum()

    def construct(self, pred, gt):
        """Construct method"""
        pos_inds = P.Select()(P.Equal()(gt, 1.0), P.Fill()(P.DType()(gt), P.Shape()(gt), 1.0), P.Fill()(P.DType()(gt),
                                                                                                        P.Shape()(gt),
                                                                                                        0.0))
        neg_inds = P.Select()(P.Less()(gt, 1.0), P.Fill()(P.DType()(gt), P.Shape()(gt), 1.0), P.Fill()(P.DType()(gt),
                                                                                                       P.Shape()(gt),
                                                                                                       0.0))

        neg_weights = self.pow(1 - gt, 4) # beta=4
        # afa=2
        pos_loss = self.log(pred) * self.pow(1 - pred, 2) * pos_inds
        neg_loss = self.log(1 - pred) * self.pow(pred, 2) * neg_weights * neg_inds

        num_pos = self.sum(pos_inds, ())
        num_pos = P.Select()(P.Equal()(num_pos, 0.0), P.Fill()(P.DType()(num_pos), P.Shape()(num_pos), 1.0), num_pos)

        pos_loss = self.sum(pos_loss, ())
        neg_loss = self.sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss

class SmoothL1LossNew(nn.Cell):
    """Smoothl1loss"""
    def __init__(self):
        super(SmoothL1LossNew, self).__init__()
        self.transpose = P.Transpose()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.shape = P.Shape()
        self.expand_dims = P.ExpandDims()
        self.sum = P.ReduceSum()
        self.cast = P.Cast()

    def construct(self, output, ind, target, wight_mask=None):
        '''
        :param output: [b, c, h, w] to [b, h, w, c]
        :param ind:
        :param target:
        :return:
        '''
        output = self.transpose(output, (0, 2, 3, 1))
        mask = P.Select()(P.Equal()(ind, 1), P.Fill()(mstype.float32, P.Shape()(ind), 1.0), P.Fill()(mstype.float32,
                                                                                                     P.Shape()(ind),
                                                                                                     0.0))
        target = self.cast(target, mstype.float32)
        output = self.cast(output, mstype.float32)
        num = self.cast(self.sum(mask, ()), mstype.float32)
        mask = self.expand_dims(mask, -1) # [batch,h,w]--[batch,h,w,c]
        output = output * mask
        target = target * mask
        loss = self.smooth_l1_loss(output, target)
        if wight_mask is not None:
            loss = loss * wight_mask
            loss = self.sum(loss, ())
        else:
            #some version need: F.depend(loss, F.sqrt(F.cast(wight_mask, mstype.float32)))
            loss = self.sum(loss, ())
        loss = loss / (num + 1e-4)
        return loss

class SmoothL1LossNewCMask(nn.Cell):
    """Smoothl1loss with mask"""
    def __init__(self):
        super(SmoothL1LossNewCMask, self).__init__()
        self.transpose = P.Transpose()
        self.smooth_l1_loss = nn.L1Loss(reduction='sum') # or use nn.SmoothL1Loss()
        self.shape = P.Shape()
        self.expand_dims = P.ExpandDims()
        self.sum = P.ReduceSum()
        self.cast = P.Cast()

    def construct(self, output, cmask, ind, target):
        '''
        :param output: [b, c, h, w] to [b, h, w, c]
        :param ind:
        :param target:
        :return:
        '''
        num = self.sum(cmask, ())
        output = self.transpose(output, (0, 2, 3, 1))

        ind = self.cast(ind, mstype.float32)
        target = self.cast(target, mstype.float32)
        cmask = self.cast(cmask, mstype.float32)
        output = self.cast(output, mstype.float32)
        ind = self.expand_dims(ind, -1)
        output = output * ind
        target = target * ind
        loss = self.smooth_l1_loss(output*cmask, target*cmask)
        #loss = self.sum(loss, ()) # if use SmoothL1Loss, this is needed
        loss = loss / (num + 1e-4)
        return loss

class CenterFaceLoss(nn.Cell):
    """
    Loss method definition.
    """
    def __init__(self, wh_weight, reg_offset, off_weight, hm_weight, lm_weight):
        super(CenterFaceLoss, self).__init__()
        # --- config parameter
        self.wh_weight = wh_weight
        self.reg_offset = reg_offset
        self.off_weight = off_weight
        self.hm_weight = hm_weight
        self.lm_weight = lm_weight
        # ---
        self.cls_loss = FocalLoss()
        self.reg_loss = SmoothL1LossNew()
        self.reg_loss_cmask = SmoothL1LossNewCMask()

    def construct(self, output_hm, output_wh, output_off, output_kps, hm, reg_mask, ind, wh, wight_mask, hm_offset,
                  hps_mask, landmarks):
        """
        Construct method.
        """
        hm_loss = self.cls_loss(output_hm, hm)  # 1. focal loss, center points
        wh_loss = self.reg_loss(output_wh, ind, wh, wight_mask)  # 2. weight and height
        off_loss = self.reg_loss(output_off, ind, hm_offset, wight_mask)  # 3. offset
        lm_loss = self.reg_loss_cmask(output_kps, hps_mask, ind, landmarks)  # 4. landmark loss

        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss + self.lm_weight * lm_loss

        # depend is needed when wight_mask and reg_mask is not been used
        F.depend(loss, F.sqrt(F.cast(wight_mask, mstype.float32)))
        F.depend(loss, F.sqrt(F.cast(reg_mask, mstype.float32)))
        # add print when you want to see loss detail and do debug
        return loss

class CenterFaceWithLossCell(nn.Cell):
    """
    Centerface with loss cell.
    """
    def __init__(self, network):
        super(CenterFaceWithLossCell, self).__init__()
        self.centerface_network = network
        self.config = config
        self.loss = CenterFaceLoss(self.config.wh_weight, self.config.reg_offset, self.config.off_weight,
                                   self.config.hm_weight, self.config.lm_weight)
        self.reduce_sum = P.ReduceSum()

    def construct(self, x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks):
        output_hm, output_wh, output_off, output_kps = self.centerface_network(x)
        loss = self.loss(output_hm, output_wh, output_off, output_kps, hm, reg_mask, ind, wh, wight_mask, hm_offset,
                         hps_mask, landmarks)
        return loss