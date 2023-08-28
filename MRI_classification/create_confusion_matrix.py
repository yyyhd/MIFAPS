import os
import json
import argparse
import sys
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pltconda
from prettytable import PrettyTable
from utils import read_split_data
from my_dataset import MyDataSet
# from model.all_resnet import resnext101_32x8d as create_model
# from model.inception_v4 import inception_v4 as create_model
# from model.all_resnet import resnet101 as create_model
from model.densenet import densenet161 as create_model
# from model.googlenet import GoogLeNet as create_model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pandas import Series, DataFrame
import pandas as pd
from util import GradCAM, show_cam_on_image, center_crop_img


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    讲解； https://blog.csdn.net/weixin_45902056/article/details/123723921
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        # print('zip(preds, labels);', zip(preds, labels))
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        print('self.matrix; ', self.matrix)
        sum_TP = 0
        for i in range(self.num_classes):
            print('self.matrix[i, i];', self.matrix[i, i])
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            print('TP;', TP)
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

class load_data(Dataset):
    def __init__(self, args):
        self.args = args
        image_path = []
        for path in glob.glob(os.path.join(args.data_path, '*.png')):
            image_path.append(path)
        self.image_path = image_path

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        path = self.image_path[idx]
        image = np.load(path, allow_pickle=True)

        patientid = image['patientid'].tolist()
        label = image['label']
        images = image['image']

        images = images.astype(np.float32)
        label = label.astype(np.float32)
        images = images[np.newaxis, ...].copy()

        return {'inputs0': images, 'label': label, 'patientid': patientid}

def main(args):
    # dataset = load_data(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")


    val_images_path, val_images_label, image_name = read_split_data(args.data_path)
    img_size = 224
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),])}
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            images_name=image_name,
                            transform=data_transform["val"])
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    print('labels', labels)
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        name = []
        scoder1 = []
        scoder0 = []
        for val_data in tqdm(val_loader, file=sys.stdout):
            images, labels, img_name = val_data
            if '_' in img_name:
                img_name = img_name.replace('_', '')
            name.append(img_name)
            outputs = model(images.to(device))
            predict = F.sigmoid(outputs).squeeze(1).cpu().detach().numpy()

            # labels = np.array(labels.cpu()) > 0
            # th = 0.5
            # predict = predict > th
            # print('labels; ', labels)
            # print('predict;', predict)
            # tp = np.sum((predict == True) & (labels == True))
            # tn = np.sum((predict == False) & (labels == False))
            # fp = np.sum((predict == True) & (labels == False))
            # fn = np.sum((predict == False) & (labels == True))
            # print('tp:{0:.4f}\ttn:{1:.4f}\tfp:{2:.4f}\tfn:{3:.4f}'.format(tp, tn, fp, fn))

            predict.tolist()
            scoder1.append(predict[0, 0])
            scoder0.append(predict[0, 1])

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

        name = DataFrame(name)
        name.columns = ['image_id']
        scoder1 = DataFrame(scoder1)
        scoder1.columns = ['0']
        scoder0 = DataFrame(scoder0)
        scoder0.columns = ['1']
        scoder = pd.concat([scoder1, scoder0], axis=1, join='inner')
        scoder = DataFrame(scoder)
        csv = pd.concat([name, scoder], axis=1, join='inner')
        csv.set_index(['image_id'], inplace=True)
        csv.to_csv(os.path.join(args.save_csv, 'test_result.csv'))

    confusion.plot()
    confusion.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./")
    parser.add_argument('--save_csv', type=str,
                        default="./")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default=',/',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
