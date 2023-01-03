# coding=utf-8
import numpy as np
import os
import re
import scipy.io as scio
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_EXTENSIONS = [
    '.JPG', '.jpg', '.JPEG', '.jpeg', '.PNG', '.png', '.PPM', '.ppm', '.BMP', '.bmp','TIF','tif'
]

# 判断文件是否为图片类型,返回True/False
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# 获取所有图片(左右图像对)
def AllImages(filepath):

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []

    # 所有图片
    for level1 in os.listdir(filepath):
        if level1.find('README') == -1:
            for level2 in os.listdir(filepath + level1 + '/'):
                if level2.find('left') > -1:
                    for level3_l in os.listdir(filepath + level1 + '/' + level2):
                        # print(list_left)
                        file_rgb = "vkitti_1.3.1_rgb/" + level1 + '/' + level2 + '/' + level3_l
                        file_disp = "vkitti_1.3.1_depthgt/" + level1 + '/' + level2 + '/' + level3_l
                        all_left_img.append(file_rgb)
                        # print(file_rgb)
                        all_left_disp.append(file_disp)
                if level2.find('right') > -1:
                    for level3_r in os.listdir(filepath + level1 + '/' + level2):
                        file_rgb = "vkitti_1.3.1_rgb/" + level1 + '/' + level2 + '/' + level3_r
                        file_disp = "vkitti_1.3.1_depthgt/" + level1 + '/' + level2 + '/' + level3_r
                        all_right_img.append(file_rgb)
                        all_right_disp.append(file_disp)
                else:
                    continue

    all_left_img.sort()
    all_right_img.sort()
    all_left_disp.sort()
    all_right_disp.sort()
    # print(all_left_img.__len__())
    # print(all_right_img.__len__())
    # print(all_left_disp.__len__())
    # print(all_right_disp.__len__())
    return all_left_img,all_right_img,all_left_disp,all_right_disp

# 得到训练集及测试集图片
def dataloader(filepath):
    all_left_img, all_right_img, all_left_disp, all_right_disp = AllImages(filepath)

    train_left_img = []
    train_right_img = []
    train_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    # 训练集图片
    for i in range(0, all_left_img.__len__()):
        if i%10!=0:
            # print(i)
            train_left_img.append(all_left_img[i])
            train_right_img.append(all_right_img[i])
            train_left_disp.append(all_left_disp[i])

    # 测试集图片
    for i in range(0, all_left_img.__len__()):
        if i%10==0:
            test_left_img.append(all_left_img[i])
            test_right_img.append(all_right_img[i])
            test_left_disp.append(all_left_disp[i])

    return train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp

# 将图片路径保存入txt中(左右图像对)
def save_to_txt(filepath,txtpath):
    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = dataloader(filepath)

    f_train = open(txtpath+"vKitti_train.txt",'w')
    for i in range(len(train_left_img)):
        f_train.write(str(train_left_img[i])+" "+str(train_right_img[i])+" "+str(train_left_disp[i])+"\n")
    f_train.close()

    f_test = open(txtpath+"vKitti_test.txt",'w')
    for j in range(len(test_left_img)):
        f_test.write(str(test_left_img[j])+" "+str(test_right_img[j])+" "+str(test_left_disp[j])+"\n")
    f_test.close()

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

# read an .pfm file into numpy array, used to load SceneFlow disparity files
def readPFM(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def aug_loss(student_ests, teacher_ests):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    aug = torch.nn.MSELoss()
    for student_est, teacher_est, weight in zip(student_ests, teacher_ests, weights):
        all_losses.append(weight * aug(student_est, teacher_est))
    return sum(all_losses)

if __name__ == '__main__':
    filepath = "/data3T_1/yuanyazhi/dataset/vKITTI/vkitti_1.3.1_rgb/"
    txtpath="/data3T_1/yuanyazhi/dataset/vKITTI/"
    save_to_txt(filepath, txtpath)

