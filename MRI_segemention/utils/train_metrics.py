import numpy as np
import os
import torch.nn as nn
import torch
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix
from skimage import filters
import torch.nn.functional as F

from utils.evaluation_metrics3D import metrics_3d, Dice


def threshold(image):
    # thresh = filters.threshold_otsu(image, nbins=256)
    # thresh = filters.threshold_li(image)
    thresh = 100
    image[image >= thresh] = 255
    image[image < thresh] = 0
    return image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    # outputs = outputs.squeeze(1)  # for MSELoss()
    # labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    Acc, SEn = 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc, sen = get_acc(img, gt)
        Acc = Acc + acc
        SEn = SEn + sen
    return Acc, SEn


def metrics3dmse(pred, label, batch_size):
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def metrics3d(pred, label, batch_size):
    pred = torch.argmax(pred, dim=1)  # for CE loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    # outputs = outputs.squeeze(1)  # for MSELoss()
    # labels = labels.squeeze(1)  # for MSELoss()
    # outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def get_acc(image, label):
    image = threshold(image)

    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    # acc = (TP + TN) / (TP + FN + TN + FP + 1e-320)
    # sen = (TP) / (TP + FN + 1e-10)
    # spe = (TN) / (TN + FP + 1e-320)
    # print('FP, FN, TP, TN;', FP, FN, TP, TN)
    return acc, sen

#####################################################################3

def MulticlassAccuracy_fn(inputs, targets, mode='eval'):
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.unsqueeze(inputs, dim=1)
    inputs = inputs.numpy()
    N = targets.shape[0]
    w, h = targets.shape[2:]

    label_values = [[0], [1], [2], [3], [4]]
    inputs = one_hot_result(inputs, label_values).astype(np.float)
    categories = targets.shape[1]

    correct_pred = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.reshape(categories, -1)
        tflat = target_.reshape(categories, -1)
        intersection = iflat * tflat
        correct_pred += intersection.sum()
    if 'train3_multidomainl_normalcl' in mode:
        accuracy = correct_pred.astype(np.float) / float(w) / float(h)
    else:
        accuracy = correct_pred.astype(np.float) / float(N)
    return accuracy


def TP_TN_FP_FN(inputs, targets, threshold=0.7):
    inputs = F.softmax(inputs, dim=1)
    # inputs_obj = inputs[:, 1, :, :]
    inputs_obj = inputs
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        TP = (iflat * tflat).sum()
        TN = ((1 - iflat) * (1 - tflat)).sum()
        FP = (iflat * (1 - tflat)).sum()
        FN = ((1 - iflat) * tflat).sum()

    acc = (TP + TN) / (TP + FN + TN + FP + 1e-320)
    sen = (TP) / (TP + FN + 1e-320)
    return sen, acc


def one_hot_result(label, label_values=[[0], [1], [2], [3], [4]]):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=1)
    return semantic_map


def metrics0(inputs, targets):
    # acc1 = MulticlassAccuracy_fn(inputs, targets, mode='eval')
    sen, acc = TP_TN_FP_FN(inputs, targets, threshold=0.5)
    
    return acc, sen