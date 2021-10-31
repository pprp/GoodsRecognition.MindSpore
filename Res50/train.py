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
"""train GENet."""
import os
import warnings
from threading import local

import mindspore.common.initializer as weight_init
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.parallel import set_algo_parameters
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, SummaryCollector,
                                      TimeMonitor)
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.args import get_args
from src.CrossEntropySmooth import CrossEntropySmooth
from src.dataset import create_dataset
from src.models import build_network
from src.install import install_pip
from src.lr_generator import get_lr
from src.utils import filter_checkpoint_parameter_by_list, str2bool, save_args


warnings.filterwarnings('ignore')
os.environ['GLOG_v'] = '3'
set_seed(1)

if __name__ == '__main__':
    # install_pip()
    args = get_args()

    print("=="*20)
    print(args)
    print("=="*20)
    save_args(args, "./checkpoints")

    if args.is_modelarts == "True":
        import moxing as mox

    device_id = 1#int(os.getenv('DEVICE_ID')) # 设置第几个gpu
    device_num = 1#int(os.getenv("RANK_SIZE"))

    ckpt_save_dir = args.save_checkpoint_path
    local_train_data_url = args.data_url

    if args.is_modelarts == "True":
        local_summary_dir = "/cache/summary"
        local_data_url = "/cache/data"
        local_train_url = "/cache/ckpt"
        local_zipfolder_url = "/cache/tarzip"
        ckpt_save_dir = local_train_url
        mox.file.make_dirs(local_train_url)
        mox.file.make_dirs(local_summary_dir)
        filename = "RP2K_rp2k_dataset.zip"
        # transfer dataset
        local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.make_dirs(local_data_url)
        local_zip_path = os.path.join(
            local_zipfolder_url, str(device_id), filename)
        mox.file.make_dirs(os.path.join(local_zipfolder_url, str(device_id)))

        obs_zip_path = os.path.join(args.data_url, filename)

        print('From Source dir: %s copy to %s' %
              (obs_zip_path, local_zip_path))

        mox.file.copy(obs_zip_path, local_zip_path)

        print("Zip file starting................")

        unzip_command = "unzip %s -d %s" % (local_zip_path, local_data_url)
        os.system(unzip_command)

        print("Zip file done.................")

        local_train_data_url = os.path.join(local_data_url, "all", "train")
        local_val_data_url = os.path.join(local_data_url, "all", "test")
    else:
        local_train_data_url = os.path.join(args.data_url, "train")
        local_val_data_url = os.path.join(args.data_url, "test")

    target = args.device_target
    # if target != 'Ascend':
    #     raise ValueError("Unsupported device target.")

    run_distribute = False

    if device_num > 1:
        run_distribute = True

    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False, device_id=device_id)

    if run_distribute:
        context.set_context(device_id=device_id,
                            enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()

    print("create training dataset.......................")

    # create dataset
    dataset = create_dataset(dataset_path=local_train_data_url, do_train=True, repeat_num=1,
                             batch_size=args.batch_size, target=target, distribute=run_distribute, image_size=args.image_size)
    step_size = dataset.get_dataset_size()

    # define net
    # mlp = str2bool(args.mlp)
    # extra = str2bool(args.extra)
    # net = net(class_num=args.class_num, extra=extra, mlp=mlp)
    # resnet50(class_num=args.class_num)
    net = build_network(args.model, args.width)

    # init weight
    if args.pre_trained:
        param_dict = load_checkpoint(args.pre_trained)

        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    print("scheduler configure................")
    # scheduler
    lr = get_lr(args.lr_init, args.lr_end,
                args.epoch_size, step_size, args.decay_mode)

    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []

    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': args.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    print("optimizer configure.................")
    opt = Momentum(group_params, lr, args.momentum,
                   loss_scale=args.loss_scale)

    print("model and loss configure................")

    # define loss, model
    if target == "Ascend":
        if not args.use_label_smooth:
            args.label_smooth_factor = 0.0

        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=args.label_smooth_factor,
                                  num_classes=args.class_num)

        loss_scale = FixedLossScaleManager(
            args.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                      metrics={'top_1_accuracy', 'top_5_accuracy'}, amp_level="O2", keep_batchnorm_fp32=False)
    elif target == "GPU":
        if not args.use_label_smooth:
            args.label_smooth_factor = 0.0

        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=args.label_smooth_factor,
                                  num_classes=args.class_num)

        loss_scale = FixedLossScaleManager(
            args.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                      metrics={'top_1_accuracy', 'top_5_accuracy'}, amp_level="O2", keep_batchnorm_fp32=False)
    else:
        raise ValueError("Unsupported device target.")

    print('callbacks configure............')
    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    rank_id = 0#int(os.getenv("RANK_ID"))
    cb = [time_cb, loss_cb]
    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_epochs*step_size,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(
            prefix=args.model, directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    dataset_sink_mode = target != "CPU"
    print("model training start....................")
    model.train(args.epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    if device_id == 0 and args.is_modelarts == "True":
        save_args(args, ckpt_save_dir)
        mox.file.copy_parallel(ckpt_save_dir, args.train_url)

    print('model evaluation start..............')
    val_dataset = create_dataset(dataset_path=local_val_data_url, do_train=False,
                                 repeat_num=1, batch_size=args.batch_size, target=target, distribute=run_distribute, image_size=args.image_size)

    res = model.eval(val_dataset)
    print("result:", res)
