"""
Evaluation metrics
"""

import numpy as np
import sklearn.metrics as metrics
import os
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
import torch
import pandas as pd
from pandas import Series, DataFrame


def DSC1(path):
    all_dsc = 0.
    file_num = 0
    for file in glob.glob(os.path.join(path, 'pred', '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        path1 = '/data/zhouheng/segemention/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=0)

        pred = pred // 255
        label = label // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        dsc = 2 * TP / (FP + 2 * TP + FN + 1e-323)
        if dsc > 0.4:
            all_dsc += dsc
            file_num += 1
        print("name:{0:.14s} ---dice: {1:.8f}".format(str(base_name), float(dsc)))
    avg_dsc = all_dsc / (file_num)
    return avg_dsc

def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def numeric_score_fov(pred, gt, mask):
    FP = np.float(np.sum((pred == 1) & (gt == 0) & (mask == 1)))
    FN = np.float(np.sum((pred == 0) & (gt == 1) & (mask == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1) & (mask == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0) & (mask == 1)))
    return FP, FN, TP, TN

def AUC(path):
    all_auc = 0.
    file_num = 0
    for file in glob.glob(os.path.join(path, '*pred.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-9] + '.png'
        # print('label_name;', label_name)
        path1 = '/data/zhouheng/mao/segmentatation/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        mask_path = '/path/to/FOV/mask/'

        pred_image = cv2.imread(file, flags=0)
        label = cv2.imread(label_path, flags=0)
        # mask = cv2.imread(mask_path, flags=-1)

        # with FOV
        label_fov = []
        pred_fov = []
        w, h = pred_image.shape
        for i in range(w):
            for j in range(h):
                # if mask[i, j] == 255:
                label_fov.append(label[i, j])
                pred_fov.append(pred_image[i, j])
        pred_image = (np.asarray(pred_fov)) / 255
        label = np.uint8((np.asarray(label_fov)) / 255)

        # pred_image = pred_image.flatten() / 255
        # label = np.uint8(label.flatten() / 255)
        # print('pred_image_shape; ', (pred_image.reshape(1, -1)).shape)
        # print('label_shape; ', (label.reshape(1, -1)).shape)

        auc_score = metrics.roc_auc_score(label.reshape(-1, 1), pred_image.reshape(-1, 1))
        all_auc += auc_score
        file_num += 1
    avg_auc = all_auc / file_num
    var_auc = np.var(all_auc)
    return avg_auc, var_auc

def DSC(path):
    all_dsc = 0.
    file_num = 0
    name = []
    dics = []
    for file in glob.glob(os.path.join(path, '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        path1 = '/data/zhouheng/mao/segmentatation/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=0)

        pred = pred // 255
        label = label // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        dsc = 2 * TP / (FP + 2 * TP + FN + 1e-323)
        all_dsc += dsc
        file_num += 1
        print("name:{0:.14s} ---dice: {1:.8f}".format(str(base_name), float(dsc)))

        name.append(label_name)
        dics.append(dsc)
        # np_dics = np.array(dics)
        # np_dics = np_dics.T
        # np.array(np_dics)
        # save = pd.DataFrame(columns = ['year', 'month'], data = [np_dics])
        # save.to_csv('/data/zhouheng/segemention/data/dics.csv',  mode='a', index=False, header=False)

    name = DataFrame(name)
    name.columns = ['name']
    dics = DataFrame(dics)
    dics.columns = ['dics']
    sum = pd.concat([name, dics], axis=1, join='inner')
    sum.set_index(['name'], inplace=True)
    sum.to_csv(os.path.join('/data/zhouheng/mao/segmentatation/data/dics.csv'))


    avg_dsc = all_dsc / (file_num)
    return avg_dsc

def AccSenSpe(path):
    all_sen = []
    all_acc = []
    all_spe = []
    for file in glob.glob(os.path.join(path, '*otsu.png')):
    # for file in glob.glob(os.path.join(path, 'pred_attu', '*pred.png')):
        base_name = os.path.basename(file)
    # print('66666666666666', file)
        label_name = base_name[:-14] + '.png'
        print('label_name', label_name)
        path1 = '/data/zhouheng/mao/segmentatation/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        # mask_path = '/path/to/FOV/mask/'

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=0)
        # mask = cv2.imread(mask_path, flags=-1)

        pred = pred // 255
        label = label // 255
        # mask = mask // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        # print('FP, FN, TP, TN', FP, FN, TP, TN)
        acc = (TP + TN) / (TP + FP + TN + FN)
        sen = TP / (TP + FN + 1e-323)
        spe = TN / (TN + FP + 1e-323)
        all_acc.append(acc)
        all_sen.append(sen)
        all_spe.append(spe)
    avg_acc, avg_sen, avg_spe = np.mean(all_acc), np.mean(all_sen), np.mean(all_spe)
    var_acc, var_sen, var_spe = np.var(all_acc), np.var(all_sen), np.var(all_spe)
    return avg_acc, var_acc, avg_sen, var_sen, avg_spe, var_spe

def FDR(path):
    all_fdr = []
    for file in glob.glob(os.path.join(path, '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        path1 = '/data/zhouheng/segemention/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=0)

        pred = pred // 255
        label = label // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        # if (FP + TP)<1e-323:
        #     continue
        fdr = FP / (FP + TP + 1e-323)   # 错误发现率
        all_fdr.append(fdr)
    return np.mean(all_fdr), np.var(all_fdr)

def dice_coeff(path):
    import torch.nn.functional as F
    threshold = 0.5
    for file in glob.glob(os.path.join(path, '*pred.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-9] + '.png'
        path1 = '/data/zhouheng/segemention/data/test/1st_manual'
        label_path = os.path.join(path1, label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=0)
        #
        # pred = Image.open(file)
        # label = Image.open(label_path).convert('L')

        inputs = pred
        targets = label

        print('type(inputs);', type(inputs))
        """Dice coeff for batches"""
        inputs = torch.from_numpy(inputs)
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :]
        inputs[inputs >= threshold] = 1
        inputs[inputs < threshold] = 0
        dice = 0.
        img_count = 0
        for input_, target_ in zip(inputs, targets):
            iflat = input_.view(-1).float()
            tflat = target_.view(-1).float()
            intersection = (iflat * tflat).sum()
            if tflat.sum() == 0:
                if iflat.sum() == 0:
                    dice_single = torch.tensor(1.0)
                else:
                    dice_single = torch.tensor(0.0)
                    img_count += 1
            else:
                dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
                img_count += 1
            dice += dice_single
    return dice


if __name__ == '__main__':
    # predicted root path
    path = '/data/zhouheng/mao/segmentatation/assets/DRIVE/pred_segnet/'
    # auc, var_auc = AUC(path)
    acc, var_acc, sen, var_sen, spe, var_spe = AccSenSpe(path)
    # fdr, var_fdr = FDR(path)
    # dice = DSC(path)
    print("sen:{0:.4f} +- {1:.4f}".format(sen, var_sen))
    print("acc:{0:.4f} +- {1:.4f}".format(acc, var_acc))
    print("spe:{0:.4f} +- {1:.4f}".format(spe, var_spe))
    # # print("fdr:{0:.4f} +- {1:.4f}".format(fdr, var_fdr))s
    # print("auc:{0:.4f}---{1:.4f}".format(auc, var_auc))
    # print("dice:{0:.4f}".format(dice))