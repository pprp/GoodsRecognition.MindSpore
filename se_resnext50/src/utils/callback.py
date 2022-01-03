from mindspore.train.callback import Callback
from src.utils.auto_mixed_precision import auto_mixed_precision


class EvalCallBack(Callback):
    def __init__(self, model, network, dataset, dataset_sink_mode, config, is_train_dataset=False):
        self.model = model
        self.network = network
        self.eval_dataset = dataset
        # epochs_to_eval是一个int数字，代表着：每隔多少个epoch进行一次验证
        self.epochs_to_eval = config.epochs_to_eval
        # self.per_eval = per_eval
        self.dataset_sink_mode = dataset_sink_mode
        self.config = config
        self.is_train_dataset = is_train_dataset

    def epoch_end(self, run_context):
        # 获取到现在的epoch数
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        # 如果达到进行验证的epoch数，则进行以下验证操作
        if cur_epoch % self.epochs_to_eval == 0:
            self.network.set_train(False)
            # 此处model设定的metrics是准确率Accuracy
            auto_mixed_precision(self.network)
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode, )
            if self.is_train_dataset:
                self.config.logger.info("-------------------训练数据集进行验证的结果-------------------")
            self.config.logger.info("------------第{}轮 top_1_accuracy: {}   top_5_accuracy: {}------------".format(
                cur_epoch, acc["top_1_accuracy"], acc["top_5_accuracy"]))

            self.network.set_train(True)