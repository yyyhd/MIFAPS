import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import scipy.misc as misc
import os
import glob
from utils.misc import thresh_OTSU, ReScaleSize, Crop
# from utils.model_eval import eval
from torchvision.transforms.functional import center_crop
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as T

DATABASE = './DRIVE/'

args = {
    'root'     : './dataset/' + DATABASE,
    # 'test_path': './dataset/' + DATABASE + 'test/',
    'test_path': '',
    'pred_path': '/./assets/' + 'DRIVE/pred_segnet/',
    'img_size' : 512
}


if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img


def ReScaleSize_DRIVE(image, re_size=512):
    w, h = image.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    image = image.crop(box)
    image = image.resize((re_size, re_size))
    return image  # , origin_w, origin_h


def ReScaleSize_STARE(image, re_size=512):
    w, h = image.size
    max_len = max(w, h)
    new_w, new_h = max_len, max_len
    delta_w = new_w - w
    delta_h = new_h - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, fill=0)
    # origin_w, origin_h = w, h
    image = image.resize((re_size, re_size))
    return image  # , origin_w, origin_h

def load_nerve():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'orig', '*.tif')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'orig', basename)
        label_name = os.path.join(args['test_path'], 'mask2', file_name + '_centerline_overlay.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels

def load_drive():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.png')):
        basename = os.path.basename(file)
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], '1st_manual', basename)
        test_images.append(image_name)
        test_labels.append(label_name)

    return test_images, test_labels

def load_stare():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.ppm')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'labels-ah', file_name + '.ah.ppm')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels

def load_padova1():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.tif')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label2', file_name + '_centerline_overlay.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels

def load_octa():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.png')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label', file_name + '_nerve_ann.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels

def load_net():
    model_path = '/data/zhouheng/mao/segmentatation/checkpoint/1Att_Net_DRIVE_200.pkl'
    # print(model_path)
    net = torch.load(model_path)
    return net

def save_prediction(pred, filename=''):
    save_path = args['pred_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    mask = pred.data.cpu().numpy() * 255
    '''
    squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    np.transpose;对图片做旋转操作
    '''
    mask = np.transpose(np.squeeze(mask, axis=0), [1, 2, 0])
    mask = np.squeeze(mask, axis=-1)
    # print('filename;', filename)
    misc.imsave(save_path + filename + '.png', mask)
    # print('已经保存！')

def dice_coeff(inputs, targets, threshold=0.5):
    import torch.nn.functional as F
    transform = T.ToTensor()  # ()很重要
    targets = transform(targets)
    """Dice coeff for batches"""
    inputs = F.softmax(inputs, dim=1)
    # inputs = inputs[:, 1, :, :]
    inputs[inputs >= threshold] = 1
    inputs[inputs < threshold] = 0
    dice = 0.
    img_count = 0
    iflat = inputs.contiguous().view(-1).float().cuda()
    tflat = targets.contiguous().view(-1).float().cuda()
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

def predict():
    net = load_net()
    # images, labels = load_nerve()
    images, labels = load_drive()
    # images, labels = load_stare()
    # images, labels = load_padova1()
    # images, labels = load_octa()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dice = 0
    j = 0
    with torch.no_grad():
        net.eval()
        for i in range(len(images)):
            print(images[i])
            name_list = images[i].split('/')
            index = name_list[-1][:-4]
            print('index;', index)
            image = Image.open(images[i])
            # image=image.convert("RGB")
            label = Image.open(labels[i])
            # label = cv2.imread(labels[i], flags=0)
            # image, label = center_crop(image, label)

            # for other retinal vessel
            # image = rescale(image)
            # label = rescale(label)
            # image = ReScaleSize_STARE(image, re_size=args['img_size'])
            # label = ReScaleSize_DRIVE(label, re_size=args['img_size'])

            # for OCTA
            # image = Crop(image)
            # image = ReScaleSize(image)
            # label = Crop(label)
            # label = ReScaleSize(label)

            # label = label.resize((args['img_size'], args['img_size']))
            # if cuda
            image = transform(image).cuda()
            # image = transform(image)
            image = image.unsqueeze(0)
            output = net(image)

            # dice0 = dice_coeff(output, label)
            # dice = dice + dice0
            # j = j+1

            save_prediction(output, filename=index + '_pred')
            print("output saving successfully")
        # print('j is= ', j)
        # print('dice;', float(dice))

if __name__ == '__main__':
    predict()
    thresh_OTSU(args['pred_path'])
