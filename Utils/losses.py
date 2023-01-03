import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2

'''total loss'''
def loss_total(loss1,loss2,loss3,a1,a2,a3):
    return loss1*a1+loss2*a2+loss3*a3

'''huber loss'''
def loss_huber(output, target, delta):
    # delta = 1
    loss = torch.where(torch.abs(target-output)<delta, 0.5*((target-output)**2),delta*torch.abs(target-output)-0.5*(delta**2))
    return torch.mean(loss)

'''log cosh loss'''
def loss_log_cosh(output, target):
    loss = torch.log(torch.cosh(output-target))
    return torch.sum(loss)

''' endpoint error '''
def loss_EPE(output, target):
  b, _, h, w = target.size()
  upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
  return torch.norm(target - upsampled_output, 1, 1).mean()

'''consistence loss'''
def loss_consistence(predict_s,predict_t):
    predict_s = F.log_softmax(predict_s)
    predict_t = F.log_softmax(predict_t)
    distance = F.pairwise_distance(predict_s, predict_t, p=2, eps=1e-06)
    return distance

'''consistence 3D loss'''
def loss_consistence_3d(predict_s,predict_t):
    # for d in d_dimension, sum
    # print(predict_s.size())
    predict_s = F.log_softmax(predict_s)
    predict_t = F.log_softmax(predict_t)
    distance = F.pairwise_distance(predict_s, predict_t, p=2, eps=1e-06)
    return distance

'''consistence 3D loss'''
def loss_discriminator_3d(predict_s,predict_t):
    # for d in d_dimension, sum
    # print(predict_s.size())
    predict_s = F.log_softmax(predict_s)
    predict_t = F.log_softmax(predict_t)
    distance = F.pairwise_distance(predict_s, predict_t, p=2, eps=1e-06)
    return distance

def model_loss_gwc(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # loss = weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)
        loss = weight * loss_huber(disp_est[mask], disp_gt[mask],2)
        # loss = weight * loss_log_cosh(disp_est[mask], disp_gt[mask])
        if loss != loss:  # loss为nan
            print("supervise nan_loss:",loss)
        all_losses.append(loss)
    return sum(all_losses)

def model_loss_gwc_warp(disp_ests, img_left, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # loss = weight * F.smooth_l1_loss(disp_est[mask], img_left[mask], size_average=True)
        loss = weight * loss_huber(disp_est, img_left,3)
        if loss != loss:  # loss为nan
            print("unsupervise nan_loss:",loss)
        all_losses.append(loss)
    return sum(all_losses)

def loss_BCE(input,target):
    return F.binary_cross_entropy(input, target, weight=None, size_average=True)


def gradient_x(img):
    gx = torch.add(img[:, :, :-1, :], -1, img[:, :, 1:, :])
    return gx


def gradient_y(img):
    gy = torch.add(img[:, :, :, :-1], -1, img[:, :, :, 1:])
    return gy


def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

    return smoothness_x + smoothness_y


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    # (input, kernel, stride, padding)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def cal_grad2_error(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta * weights_x[:, :, :, 1:] * torch.abs(dx2)) + torch.mean(
        beta * weights_y[:, :, 1:, :] * torch.abs(dy2))) / 2.0


def warp_2(est, img, occ_mask, args):
    l1_warp2 = torch.abs(est - img) * occ_mask
    l1_reconstruction_loss_warp2 = torch.mean(l1_warp2) / torch.mean(occ_mask)
    ssim_warp2 = SSIM(est * occ_mask, img * occ_mask)
    ssim_loss_warp2 = torch.mean(ssim_warp2) / torch.mean(occ_mask)
    image_loss_warp2 = args.alpha_image_loss * ssim_loss_warp2 + (
                1 - args.alpha_image_loss) * l1_reconstruction_loss_warp2
    return image_loss_warp2


def create_mask(tensor, paddings):
    shape = tensor.shape
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner = Variable(torch.ones((inner_height, inner_width)).cuda())

    mask2d = nn.ZeroPad2d((paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]))(inner)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()


def create_border_mask(tensor, border_ratio=0.1):
    num_batch, _, height, width = tensor.shape
    sz = np.ceil(height * border_ratio).astype(np.int).item(0)
    border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()


def length_sq(x):
    return torch.sum(x ** 2, 1, keepdim=True)