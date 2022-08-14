import torch
from torch import nn

import constants


class MultiLabelCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiLabelCrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros(y_true.size(0), 1, dtype=torch.float, device=constants.DEVICE)
        y_pred_neg = torch.cat((y_pred_neg, zeros), dim=1)
        y_pred_pos = torch.cat((y_pred_pos, zeros), dim=1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=1)
        return (neg_loss + pos_loss).mean()
