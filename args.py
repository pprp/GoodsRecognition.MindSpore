import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('--config', type=str,
                        default="./config/default.yaml", help='yaml files')
    parser.add_argument('--data_url', type=str,
                        default='/home/pdluser/dataset/sub/train', help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None,
                        help='Train output path')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                        help="Device target, support Ascend, GPU and CPU.")
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='Pretrained checkpoint path')
    parser.add_argument('--extra', type=str, default="True",
                        help='whether to use Depth-wise conv to down sample')
    parser.add_argument('--mlp', type=str, default="True",
                        help='bottleneck . whether to use 1*1 conv')
    parser.add_argument('--is_modelarts', type=str,
                        default="False", help='is train on modelarts')
    args = parser.parse_args()

    # process argparse & yaml
    if not args.config:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:  # yaml priority is higher than args
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = argparse.Namespace(**opt)

    return args
