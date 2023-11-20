import tensorboardX
import torch
import numpy as np
import glob
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score


class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def cal_acc(outputs, gt, threshold=0.5):
    outputs = np.array(outputs > threshold, dtype=outputs.dtype)

    correct_pred = (np.logical_and(outputs, gt)).astype(np.float64).flatten()
    acc = correct_pred.sum() / gt.sum()
    return acc


def cal_auc(outputs, gt):
    batch_size = outputs.shape[0]
    auc = 0
    for idx in range(batch_size):
        trans_output = outputs[idx].flatten()
        trans_gt = gt[idx].flatten().astype(np.int32)
        try:
            roc_auc = roc_auc_score(trans_gt, trans_output)
        except ValueError:
            trans_gt[0] = (trans_gt[0] + 1) % 2
            trans_output[0] = 1 - trans_output[0]
            roc_auc = roc_auc_score(trans_gt, trans_output)
            # roc_auc = 0
            # pass  ##或者其它定义，例如roc_auc=0
        auc += roc_auc
    auc = auc / batch_size

    return auc


def cal_dice(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(y_pred, y_truth, eps=1e-6):
    batch_size = y_truth.size(0)
    y_pred = y_pred.type(torch.FloatTensor)
    dice = 0.
    for i in range(batch_size):
        intersection = torch.sum(torch.mul(y_pred[i], y_truth[i])) + eps / 2
        union = torch.sum(y_pred[i]) + torch.sum(y_truth[i]) + eps
        dice += 2 * intersection / union
    return dice / batch_size


def f1_score(y_true, y_pred):
    e = 1e-8
    gp = np.sum(y_true)
    tp = np.sum(y_true * y_pred)
    pp = np.sum(y_pred)
    p = tp / (pp + e)
    r = tp / (gp + e)
    f1 = (2 * p * r) / (p + r + e)
    return f1


def cal_f1(outputs, gt, threshold):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return:
    """
    batch_size = outputs.shape[0]
    f1 = 0
    outputs = np.array(outputs > threshold, dtype=outputs.dtype)
    for idx in range(batch_size):
        trans_output = outputs[idx].flatten()
        trans_gt = gt[idx].flatten()
        f1 += f1_score(trans_gt, trans_output)
    f1 = f1 / batch_size

    return f1


def cal_iou(outputs, gt, threshold):
    """IoU = intersection(A, B) / union(A, B)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return: IoU
    """
    eps = 1e-8
    batch_size = outputs.shape[0]
    iou = 0
    outputs = np.array(outputs > threshold, dtype=outputs.dtype)
    for idx in range(batch_size):
        intersection = np.logical_and(outputs[idx], gt[idx])
        union = np.logical_or(outputs[idx], gt[idx])
        intersection = np.sum(intersection)
        union = np.sum(union)
        iou += intersection / (union + eps)
    iou = iou / batch_size
    return iou


def cal_fnr(premask, groundtruth):
    premask = np.array(premask > 0.5, dtype=premask.dtype)
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    fnr = false_neg / (true_pos + false_neg)

    return fnr


def calculate_metric_score(outputs, targets, threshold=0.5, metric_name=None):
    auc = cal_auc(outputs, targets)
    f1 = cal_f1(outputs, targets, threshold)
    iou = cal_iou(outputs, targets, threshold)

    if metric_name == 'auc':
        return auc
    elif metric_name == 'f1':
        return f1
    elif metric_name == 'iou':
        return iou
    elif metric_name == None:
        return auc, f1, iou



