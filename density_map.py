# -*- coding: utf-8 -*-
# @Time    : 2023.12.2
# @Author  : jie
# @File    : density_map.py
import glob
from tqdm import tqdm
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from scipy.io import loadmat
from matplotlib import cm
import numpy as np
import os


def generate_density(img, points):
    """
    Generate density map from points.
    :param img: image
    :param points: points
    :return:  dot map and density map
    """
    density = np.zeros(img.shape[:2], dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density
    # 建立一个KDTree，用于快速查找最近邻点
    kdtree = scipy.spatial.KDTree(points.copy(), leafsize=2048)
    # 查询kdtree,返回包括自身在内的最近4个点的距离和位置
    distances, locations = kdtree.query(points, k=4)
    pt2d_sum = np.zeros(img.shape[:2], dtype=np.float32)
    for i, p in enumerate(points):
        pt2d = np.zeros(img.shape[:2], dtype=np.float32)
        pt2d[int(p[1]), int(p[0])] = 1.
        pt2d_sum += pt2d
        if gt_count > 1:
            # 最近3个点的距离的平均值
            average_distance = (distances[i][1] + distances[i][2] + distances[i][3])/3.
            # 论文中根据经验发现 β = 0.3 给出了最好的结果
            sigma = average_distance * 0.3
            # print("sigma:", sigma)
        else:
            # why?这样设置sigma的值比较大，会使得生成的高斯核比较大，从而使得生成的密度图比较模糊
            sigma = np.average(np.array(img.shape[:2])) / 2. / 2.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return pt2d_sum, density


# 生成密度图
if __name__ == '__main__':
    root = r'.\Dataset\ShanghaiTech\part_A\test_data'
    img_path = os.path.join(root, r'images')
    gt_path = os.path.join(root, r'ground_truth')
    img_paths = []
    for i in glob.glob(os.path.join(img_path, '*.jpg')):
        img_paths.append(i)
    # print(img_paths, len(img_paths))
    for img_p in tqdm(img_paths, desc=f'生成密度图', total=len(img_paths)):
        img = plt.imread(img_p)
        mat = loadmat(img_p.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat'))
        points = mat['image_info'][0, 0][0, 0][0]
        _, gt = generate_density(img, points)
        np.save(img_p.replace('images', 'ground_truth').replace('.jpg', '.npy'), gt)




#
# test code
# if __name__ == '__main__':
#     root = r'.\Dataset\ShanghaiTech'
#
#     train_images = os.path.join(root, r'part_A\train_data\images')
#     img_path = os.path.join(train_images, r'IMG_294.jpg')
#     img = plt.imread(img_path)
#     mat = loadmat(img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat'))
#
#     points = mat['image_info'][0, 0][0, 0][0]
#
#     point_map, gt = generate_density(img, points)
#
#     # 设置显示的图片大小
#     plt.figure(figsize=(12, 6))
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(img)
#     plt.title('test image')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(point_map, cmap='gray')
#     plt.title('Dot map')
#     '''
#        "jet" 是一种从蓝色到红色的渐变颜色映射，常用于表示温度、速度等连续变化的数据。
#        例如，低数值可能显示为蓝色，中间值为绿色，高数值为红色。
#        '''
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(gt, cmap='jet')
#     plt.title('Density map')
#     plt.show()
