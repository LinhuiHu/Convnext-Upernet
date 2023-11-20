import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F


class BCEDicedLoss(_Loss):
    def __init__(self, bce_weight=1.0):
        super(BCEDicedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        combined_loss = dice_loss + self.bce_weight * bce_loss

        return combined_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):
        batch = target.size(0)
        input_flat = output.view(batch, -1)
        target_flat = target.view(batch, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / batch
        return loss