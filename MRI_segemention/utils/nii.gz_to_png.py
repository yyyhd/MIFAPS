# 周恒
import os
import numpy as np
import nibabel as nib
import imageio
import matplotlib
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pylab as plt

def read_niifile(niifilepath):  # 读取niifile文件
    img = nib.load(niifilepath)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    return img_fdata

def save_figx(niifilepath, savepath):  # 输出x方向的图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(x):
        silce = fdata[k, :, :]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式

def save_figy(niifilepath, savepath):  # 输出y方向的图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(y):
        silce = fdata[:, k, :]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式

def save_figz(niifilepath, savepath):  # 输出z方向的图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(z):
        silce = fdata[:, :, k]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式

if __name__ == '__main__':
    niifilepath = 'D:\\a file who should been E\\示例数据\\lvsili_label_tooth.nii.gz'
    savepath = 'D:\\a file who should been E\\示例数据\\x'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save_figz(niifilepath, savepath)