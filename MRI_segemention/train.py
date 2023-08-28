"""
Training script for CS-Net
"""
print('6666666666')
from torch.nn import functional as F
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
# from model.csnet import CSNet
from model.csnet import CSNet
# from model.csnet import CSNet
from dataloader.drive import Data
from utils.train_metrics import metrics0, metrics
from utils.visualize import init_visdom_line, update_lines
from utils.dice_loss_single_class import dice_coeff_loss, dice_coeff, CrossEntropyLoss2d, MulticlassDiceLoss, CEMDiceLoss, diceloss2d
from torchvision.utils import save_image
import argparse
import ast
from utils.criterion import SegmentationLoss
import logging
# from visdom import Visdom
import matplotlib.pyplot as plt
from torchnet import meter
import torch
from torchvision import datasets
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

args = {
    'root'      : '',
    'data_path' : '',
    'epochs'    : 200,
    'lr'        : 0.0001,
    'snapshot'  : 1,
    'test_step' : 1,
    'ckpt_path' : '',
    'batch_size': 8,
    'ce_weight' : 1.0,
    'save_path' :'',
    'loss_path' :'./epoch_{}',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for CHAOS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='fuseunet', type=str, help='fuseunet, ...')
    parser.add_argument('--data_mean', default=None, nargs= '+', type=float,
                        help='Normalize mean')
    parser.add_argument('--data_std', default=None, nargs= '+', type=float,
                        help='Normalize std')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--loss', default='cedice', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=256, type=int, help='512')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--cedice_weight', default=[1.0, 1.0], nargs= '+', type=float,
                        help='weight for ce and dice loss')
    parser.add_argument('--ceclass_weight', default=[1.0, 1.0], nargs= '+', type=float,
                        help='categorical weight for ce loss')
    parser.add_argument('--diceclass_weight', default=[1.0, 1.0], nargs= '+', type=float,
                        help='categorical weight for dice loss')
    parser.add_argument('--checkpoint', default='checkpoint_chaos_comparison1case/')
    parser.add_argument('--history', default='history_chaos_comparison1case')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--repetition', default=2, type=int, help='...')
    parser.add_argument('--tb_dir', default='/data/zhouheng/segemention/tb_dir')



    args = parser.parse_args()
    return args

# # Visdom---------------------------------------------------------
# X, Y = 0, 0.5  # for visdom
# x_acc, y_acc = 0, 0
# x_sen, y_sen = 0, 0
# env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
# env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
# env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
# # ---------------------------------------------------------------

def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'Att_Net_DRIVE_' + str(iter) + '.pkl')
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_loss(n):
    y = []
    y1 = []
    y2 = []
    for i in range(0, 200):
        enc = np.load('/epoch_{}.npy'.format(i))
        tempy = list(enc)
        tempy = sum(tempy)/len(tempy)
        y.append(tempy)
    x = range(0, len(y))
    # for i in range(50, n):
    #     enc = np.load('/media/data/zhouheng/mao/calss/Loss/1/loss_resnet/epoch_{}.npy'.format(i))
    #     tempy = list(enc)
    #     tempy = sum(tempy)/len(tempy)
    #     y1.append(tempy)
    # x1 = range(0, len(y1))
    # for i in range(50, n):
    #     enc = np.load('/media/data/zhouheng/mao/calss/Loss/1/loss_resnext/epoch_{}.npy'.format(i))
    #     tempy = list(enc)
    #     tempy = sum(tempy)/len(tempy)
    #     y2.append(tempy)
    # x2 = range(0, len(y2))

    plt.plot(x, y)
    # plt.plot(x1, y1, label='ResNet101')
    # plt.plot(x2, y2, label='ResNet101')
    plt_title = 'BATCH_SIZE = 8; LEARNING_RATE:0.0001'
    plt.title(plt_title)
    plt.xlabel('per 200 times')
    plt.ylabel('LOSS')
    plt.rcParams['font.size'] = 8
    plt.legend(fontsize=8)

    file_name = '/...'
    plt.savefig(file_name)
    # plt.show()


def train(arg):
    # set the channels to 3 when the format is RGB, otherwise 1.
    cedice_weight = torch.tensor(arg.cedice_weight)
    ceclass_weight = torch.tensor(arg.ceclass_weight)
    diceclass_weight = torch.tensor(arg.diceclass_weight)
    if arg.loss == 'ce':
        criterion = CrossEntropyLoss2d(weight=ceclass_weight).cuda()
    elif arg.loss == 'dice':
        criterion = MulticlassDiceLoss(weight=diceclass_weight).cuda()
    elif arg.loss == 'cedice':
        criterion = CEMDiceLoss(cediceweight=cedice_weight, ceclassweight=ceclass_weight,
                                diceclassweight=diceclass_weight).cuda()

    # net = UNet(classes=1, channels=3).cuda()
    net = CSNet(classes=1, channels=3).cuda()
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0001)
    # critrion = nn.MSELoss().cuda()
    critrion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
    # critrion = nn.CrossEntropyLoss().cuda()
    print("---------------start training------------------")
    # load train dataset
    train_data = Data(args['data_path'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=0, shuffle=False)

    iters = 1
    accuracy = 0.
    sensitivty = 0.

    for epoch in range(args['epochs']):
        net.train()
        save_Loss = []
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            # pred = net(image)['out']  # FCN
            # pred = F.sigmoid(pred)    # FCN
            pred = net(image)
            loss1 = diceloss2d(pred, label)
            loss2 = dice_coeff_loss(pred, label)
            loss = loss1*0.5+loss2*0.5
            running_loss = loss.item()
            loss.backward()
            optimizer.step()
            acc, sen = metrics(pred, label, pred.shape[0])

            # save pred images
            if idx % 2 == 0:
                _image = image[0]
                _label = label[0]
                _out_image = pred[0]

                img = torch.stack([_label, _out_image], dim=0)
                save_path = args['save_path']
                save_image(img, f'{save_path}/{idx}.png')


            # # # ---------------------------------- visdom --------------------------------------------------
            save_Loss.append(running_loss)
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}'.format(epoch + 1,
                                                                                 iters, loss.item(),
                                                                                 acc / 8,
                                                                                 sen / 8))
            iters += 1

        Loss0 = np.array(save_Loss)
        np.save(args['loss_path'].format(epoch), Loss0)

        # save model
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1)

        # # model eval
        # if (epoch + 1) % args['test_step'] == 0:
        #     test_acc, test_sen = model_eval(net)
        #     print("Average acc:{0:.4f}, average sen:{1:.4f}".format(test_acc, test_sen))
        #
        #     if (accuracy > test_acc) & (sensitivty > test_sen):
        #         save_ckpt(net, epoch + 1 + 8888888)
        #         accuracy = test_acc
        #         sensitivty = test_sen


def model_eval(net):
    print("Start testing model...")
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc, Sen = [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc, sen = metrics(pred_val, label, pred_val.shape[0])
        print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}".format(acc, sen))
        Acc.append(acc)
        Sen.append(sen)
        file_num += 1
        # for better view, add testing visdom here.
        return np.mean(Acc), np.mean(Sen)


if __name__ == '__main__':
    # arg = parse_args()
    plot_loss(649)
    # train(arg)
