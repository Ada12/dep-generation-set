from torch import nn

from hyper_params import HyperParams
from module.multilabel_crossentropy import MultiLabelCrossEntropy


def loss_func(opts: HyperParams):
    if opts.loss_func == 'ce':
        return nn.BCEWithLogitsLoss()
    elif opts.loss_func == 'mce':
        return MultiLabelCrossEntropy()
