import os
import sys
import json
import pickle
import random
import torch.nn.functional as F
import torch
from tqdm import tqdm
from pandas import Series, DataFrame
import pandas as pd
from torchvision import transforms

import matplotlib.pyplot as plt


def read_split_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    patient_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    print('patient_class;', patient_class)
    # 排序，保证顺序一致
    patient_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(patient_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    print('json;', json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []   # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    train_name = []
    every_class_num = []     # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in patient_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        # 按比例随机采样验证样本
        for img_path in images:
            image_name = os.path.basename(img_path)
            train_name.append(image_name[:-8])
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, train_name


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    save_csv = '/data/zhouheng/mao/calss/densenet'
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    name_list = []
    scoder1 = []
    scoder0 = []
    for step, data in enumerate(data_loader):
        images, labels, img_name = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        predict = F.sigmoid(pred).squeeze(1).cpu().detach().numpy()
        pred_classes = torch.max(pred, dim=1)[1]

        for i in range(0, len(img_name)):
            name_list.append(img_name[i])
            scoder0.append(predict[i, 0])
            scoder1.append(predict[i, 1])

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    name_list = DataFrame(name_list)
    name_list.columns = ['image_id']
    scoder1 = DataFrame(scoder1)
    scoder1.columns = ['1']
    scoder0 = DataFrame(scoder0)
    scoder0.columns = ['0']
    scoder = pd.concat([scoder1, scoder0], axis=1, join='inner')
    scoder = DataFrame(scoder)
    csv = pd.concat([name_list, scoder], axis=1, join='inner')
    csv.set_index(['image_id'], inplace=True)
    # print('step', epoch)
    # if epoch//10 == 0:
    csv.to_csv(os.path.join(save_csv, 'train_result-{}.csv'.format(epoch)))


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num