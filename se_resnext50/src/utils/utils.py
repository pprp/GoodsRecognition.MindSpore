import datetime
import os
import time

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.train.callback import Callback
from src.config import config
from src.utils.logging import get_logger
from src.utils.data_prepare import normal_process

def modelarts_process(config):
    print(f"enable modelarts: {config.enable_modelarts}")
    if config.enable_modelarts:
        print("modelarts_process begin.....")

        config.device_id = int(os.getenv("DEVICE_ID"))
        config.device_num = int(os.getenv("RANK_SIZE"))
        import moxing as mox

        local_summary_dir = "/cache/summary"
        local_data_url = "/cache/data"
        local_train_url = "/cache/ckpt"
        local_zipfolder_url = "/cache/tarzip"
        mox.file.make_dirs(local_train_url)
        mox.file.make_dirs(local_summary_dir)
        filename = "RP2K_rp2k_dataset.zip"
        # transfer dataset
        local_data_url = os.path.join(local_data_url, str(config.device_id))
        mox.file.make_dirs(local_data_url)
        local_zip_path = os.path.join(
            local_zipfolder_url, str(config.device_id), filename
        )
        mox.file.make_dirs(os.path.join(local_zipfolder_url, str(config.device_id)))

        obs_zip_path = os.path.join(config.data_url, filename)

        print("From Source dir: %s copy to %s" % (obs_zip_path, local_zip_path))

        mox.file.copy(obs_zip_path, local_zip_path)

        print("Zip file starting................")

        unzip_command = "unzip %s -d %s" % (local_zip_path, local_data_url)
        os.system(unzip_command)

        print("Zip file done.................")

        config.data_path = os.path.join(local_data_url, "all", "train")
        config.eval_data_path = os.path.join(local_data_url, "all", "test")

        print("preprocess dataset.......")

        normal_process(config)
        
        print("preprocess dataset done .......")

        config.device_target = "Ascend"
    else:
        normal_process(config)


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
        self.args.logger.info("start network train...")

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = (
            self.args.per_batch_size
            * (me_step - self.me_epoch_start_step_num)
            * self.args.group_size
            / time_used
        )
        self.args.logger.info(
            "epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f}"
            "imgs/sec".format(real_epoch, me_step, cb_params.net_outputs, fps_mean)
        )

        if self.args.rank_save_ckpt_flag:
            import glob

            ckpts = glob.glob(os.path.join(self.args.outputs_dir, "*.ckpt"))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith("{}-".format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info(
                    "epoch[{}], iter[{}], loss:{}, ckpt:{},"
                    "ckpt_fn:{}".format(
                        real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn
                    )
                )

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info("end network train...")


def set_parameters(config):
    """parameters"""
    context.set_context(
        mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id = config.device_id
    )

    if config.run_distribute:
        context.set_context(device_id=config.device_id,
                            enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(device_num=config.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()

    # init distributed
    if config.run_distribute:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    if config.is_dynamic_loss_scale == 1:
        config.loss_scale = (
            1  # for dynamic loss scale can not set loss scale in momentum opt
        )

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(
        config.output_path, datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S")
    )
    config.logger = get_logger(config.outputs_dir, config.rank)
    return config
