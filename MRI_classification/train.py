import os
import argparse
# from torchvision import models
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from model.densenet import densenet161 as create_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.nn as nn
import numpy as np


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, img_name = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                      ])}
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              images_name=img_name,
                              transform=data_transform["train"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    # model = models.resnet101(pretrained=True)
    # net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # assert os.path.exists(args.weights), "file {} does not exist.".format(args.weights)
    # model.load_state_dict(torch.load(args.weights, map_location=device), False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # change fc layer structure
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, 2)
    # model.to(device)


    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     # print(model.load_state_dict(weights_dict, strict=False))
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "head" not in name:
    #             para.requires_grad_(False)
    #             # print("training {}".format(name))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        save_Loss = []
        # train
        #####---自动更新学习率的优化器---#####
        # LEARNING_RATE = max(lr * (0.1 ** (epoch // 100)), 1e-5)
        # optimizer = torch.optim.SGD([
        #     {'params': model.parameters()}
        # ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)


        tags = ["train_loss", "train_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "/.../model-{}.pth".format(epoch))

        running_loss = train_loss
        save_Loss.append(running_loss)
        Loss0 = np.array(save_Loss)
        np.save(args.loss_path.format(epoch), Loss0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=650)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./...")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--loss_path', type=str,
                        default="./epoch_{}")

    opt = parser.parse_args()

    main(opt)
