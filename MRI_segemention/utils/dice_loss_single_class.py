import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image



def dice_coeff(inputs, targets, threshold=0.5):
    """Dice coeff for batches"""
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


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        if len(targets.shape)>3:
            targets = torch.argmax(targets.float(), dim=1)
        return self.crossentropy_loss(inputs, targets)
    
    
class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0, reduction='mean'):
        super(MulticlassDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction
        self.dice = DiceLoss(smooth=self.smooth, reduction=self.reduction)

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        totalLoss = 0
        if len(target.shape)>3:
            C = target.shape[1]
            for i in range(C):
                diceLoss = self.dice(input[:,i], target[:,i])
                if self.weight is not None:
                    diceLoss *= self.weight[i]
                totalLoss += diceLoss
        else:
            totalLoss = self.dice(input[:,1], target)
        return totalLoss
    
    
class CEMDiceLoss(nn.Module):
    def __init__(self, cediceweight=None, ceclassweight=None, diceclassweight=None, reduction='mean'):
        super(CEMDiceLoss, self).__init__()
        self.cediceweight = cediceweight
        self.ceclassweight = ceclassweight
        self.diceclassweight = diceclassweight
        self.ce = CrossEntropyLoss2d(weight=ceclassweight, reduction=reduction)
        self.multidice = MulticlassDiceLoss(weight=diceclassweight, reduction=reduction)

    def forward(self, inputs, targets):
        save_path = '/data/zhouheng/segemention/data/train_image'
        save_image(targets, f'{save_path}/{20}.png')
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.multidice(inputs, targets)
        if self.cediceweight is not None:
            loss = ce_loss * self.cediceweight[0] + dice_loss * self.cediceweight[1]
        else:
            loss = ce_loss + dice_loss
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        N = target.size(0)
        if len(input.shape) > 3:
            input = F.softmax(input, dim=1)
            iflat = input[:,1,:,:].view(N, -1).float()
        else:
            iflat = input.view(N, -1).float()
        tflat = target.view(N, -1).float()
        intersection = iflat*tflat
        loss = 1.0 - (2. * intersection.sum(1) + self.smooth) / \
               (iflat.sum(1) + tflat.sum(1) + self.smooth)
        if self.reduction == 'mean':
            diceloss = loss.sum() / N
        elif self.reduction == 'sum':
            diceloss = loss.sum()
        elif self.reduction == 'none':
            diceloss = loss
        else:
            print('Wrong')
        return diceloss


def diceloss2d(inputs, targets, smooth=1.0, reduction='mean'):
    inputs_obj = F.softmax(inputs, dim=1)
    # inputs_obj = F.softmax(inputs)
    # inputs_obj = inputs[:, 1, :, :]
    diceloss = []

    iflat = inputs_obj.view(inputs.shape[0], -1).float()
    tflat = targets.view(targets.shape[0], -1).float()
    intersection = torch.sum(iflat*tflat, dim=1)
    loss = 1.0 - (((2. * intersection + smooth) / (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + smooth)))

    if reduction == 'mean':
        diceloss = loss.sum() / inputs.shape[0]
    elif reduction == 'sum':
        diceloss = loss.sum()
    elif reduction == 'none':
        diceloss = loss
    else:
        print('Wrong')
    return diceloss

#########################上面新加
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        # target = _make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 0.0001
        # dot是返回两个矩阵的点集
        # inter,uniun:两个值的大小分别是10506.6,164867.2
        # print('input.view(-1); ', input.view(-1))
        # print('target.view(-1); ', target.view(-1))
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        # 这里没有打印出来，难道没有执行到这里吗
        # print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff_loss(input, target):
    return 1 - dice_coeff(input, target)



def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)