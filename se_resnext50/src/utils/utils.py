import datetime
import os
import time

import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.train.callback import Callback
from src.model_utils.config import config
from src.utils.logging import get_logger


class BuildTrainNetwork(nn.Cell):
    """build training network"""
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class ProgressMonitor(Callback):
    """monitor loss and time"""
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args
        self.ckpt_history = []

    def begin(self, run_context):
        self.args.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.args.per_batch_size * (me_step-self.me_epoch_start_step_num) * self.args.group_size / time_used
        self.args.logger.info('epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f}'
                              'imgs/sec'.format(real_epoch, me_step, cb_params.net_outputs, fps_mean))

        if self.args.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.args.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{},'
                                      'ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))


        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info('end network train...')

def set_parameters():
    """parameters"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False)
    # init distributed
    if config.run_distribute:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    if config.is_dynamic_loss_scale == 1:
        config.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.output_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    return config
