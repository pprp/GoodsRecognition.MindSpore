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
"""train ImageNet."""
import os

from mindspore import Tensor, context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.loss_scale_manager import (
    DynamicLossScaleManager,
    FixedLossScaleManager,
)
from mindspore.train.model import Model
import mindspore.nn as nn

from src.config import config
from src.models import build_network
from src.py_dataset import classification_dataset

# from src.image_classification import get_network
from src.model_utils.moxing_adapter import moxing_wrapper
from src.utils.callback import EvalCallBack
from src.losses import build_loss
from src.utils.lr_generator import get_lr
from src.utils.optimizer_param import get_param_groups
from src.utils.utils import set_parameters, ProgressMonitor, modelarts_process
from src.utils.var_init import load_pretrain_model
from src.utils.controller import build_lr_scheduler, build_optimizer
set_seed(1)
from src.losses.arcface import MyNetWithLoss 


def train():
    modelarts_process(config)
    """training process"""
    set_parameters(config)
    if int(os.getenv("DEVICE_ID", "0")):
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))

    # init distributed
    if config.run_distribute:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(
            parallel_mode=parallel_mode,
            device_num=config.group_size,
            gradients_mean=True,
        )

    print(f"load train data from {config.data_path}; \n load test data from {config.eval_data_path}")
    # dataloader
    de_dataset = classification_dataset(
        config.data_path,
        config.image_size,
        config.per_batch_size,
        1,
        config.rank,
        config.group_size,
        num_parallel_workers=8,
        config=config,
    )
    config.steps_per_epoch = de_dataset.get_dataset_size()

    val_dataset = classification_dataset(
        config.eval_data_path,
        config.image_size,
        config.per_batch_size,
        1,
        config.rank,
        config.group_size,
        num_parallel_workers=8,
        mode="others",
        config=config,
    )

    config.logger.save_args(config)

    # network
    config.logger.important_info("start create network")

    print(f"build network {config.model_name}")
    
    # get network and init
    network = build_network(config.model_name, config.num_classes)

    net = MyNetWithLoss(network, config)

    load_pretrain_model(config.checkpoint_file_path, network, config)

    # lr scheduler
    lr = build_lr_scheduler(config)

    # optimizer
    opt = build_optimizer(config, lr, network)

    # loss
    loss = build_loss(config=config)

    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(
            init_loss_scale=65536, scale_factor=2, scale_window=2000
        )
    else:
        loss_scale_manager = FixedLossScaleManager(
            config.loss_scale, drop_overflow_update=False
        )

    model = Model(
        net,
        loss_fn=loss,
        optimizer=opt,
        loss_scale_manager=loss_scale_manager,
        metrics={"top_1_accuracy", "top_5_accuracy"},
        amp_level="O2",# GPU O2 Ascend O3
    )

    # checkpoint save
    progress_cb = ProgressMonitor(config)
    eval_cb = EvalCallBack(
        model,
        network,
        val_dataset,
        dataset_sink_mode=False,
        config=config,
        is_train_dataset=False,
    )
    train_cb = EvalCallBack(
        model,
        network,
        de_dataset,
        dataset_sink_mode=False,
        config=config,
        is_train_dataset=True,
    )

    callbacks = [progress_cb, eval_cb, train_cb]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
            keep_checkpoint_max=config.ckpt_save_max,
        )
        save_ckpt_path = os.path.join(
            config.outputs_dir, "ckpt_" + str(config.rank) + "_model_" + config.model_name + "/"
        )
        ckpt_cb = ModelCheckpoint(
            config=ckpt_config,
            directory=save_ckpt_path,
            prefix="{}_model_name_{}".format(config.rank, config.model_name),
        )
        callbacks.append(ckpt_cb)

    print("traininng begin .......")
    model.train(
        config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=True
    )

    if config.enable_modelarts:
        import moxing as mox 
        if config.device_id == 0:
            mox.file.copy_parallel(config.outputs_dir, config.train_url)


if __name__ == "__main__":
    train()
