# -*- coding: utf-8 -*-
# @Time    : 2023.12.2
# @Author  : jie
# @File    : dataset.py
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

#
# dataset类
class CrowdDataset(Dataset):
    def __init__(self, img_root, gt_map_root, gt_down_samples=1):
        """
        :param img_root: 图片路径
        :param gt_map_root: ground truth路径
        :param gt_down_samples: ground truth down_sample
        """
        self.img_root = img_root
        self.gt_map_root = gt_map_root
        self.gt_down_samples = gt_down_samples
        # 获取图片路径下的所有图片名
        self.img_names = [filename for filename in os.listdir(img_root) \
                          if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # 断言判断index是否在范围内
        assert index <= len(self), 'index range error'
        # 获取图片名
        img_name = self.img_names[index]
        # 读取图片,以及对应的标注数据
        img = plt.imread(os.path.join(self.img_root, img_name))
        gt_map = np.load(os.path.join(self.gt_map_root, img_name.replace('.jpg', '.npy')))
        # 如果图片是灰度图，则将其转换为三通道
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)
        # 如果gt_down_samples大于1，则需要对图片和标注数据进行下采样
        if self.gt_down_samples > 1:
            ds_rows = int(img.shape[0] // self.gt_down_samples)
            ds_cols = int(img.shape[1] // self.gt_down_samples)
            # resize函数接受的是(cols,rows)，即第一个参数是宽，第二个参数是高
            img = cv2.resize(img, (ds_cols * self.gt_down_samples, ds_rows * self.gt_down_samples))
            # 转换为(channel,rows,cols)
            img = img.transpose((2, 0, 1))
            gt_map = cv2.resize(gt_map, (ds_cols, ds_rows))
            # 这一步为什么需要乘以self.gt_down_samples*self.gt_down_samples？
            gt_map = gt_map[np.newaxis, :, :] * self.gt_down_samples * self.gt_down_samples
        # 转换为tensor数据格式
        img_tensor = torch.tensor(img, dtype=torch.float)
        gt_map_tensor = torch.tensor(gt_map, dtype=torch.float)
        return img_tensor, gt_map_tensor


# 测试代码
if __name__ == '__main__':
    img_root = r'D:\project\py_prj\MCNN-pytorch\Dataset\ShanghaiTech\part_A\train_data\images'
    gt_map_root = img_root.replace('images', 'ground_truth')

    dataset = CrowdDataset(img_root, gt_map_root, gt_down_samples=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print('length of dataloader:', len(dataloader), '\n')
    for i ,(img_tensor, gt_map_tensor) in enumerate(dataloader):
        print(i, img_tensor.shape, gt_map_tensor.shape)
