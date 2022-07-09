# coding=utf-8
import scipy.io as scio
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import os
import scipy.misc as misc
import random
from torch.utils.data import Dataset
from Datasets.data_io import get_transform, read_all_lines, get_transform1

class UnderwaterDataset(Dataset):
    def __init__(self, datapath_rgb, datapath_d, list_filename, training, scaling):
        self.datapath_rgb = datapath_rgb
        self.datapath_d = datapath_d
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.scaling = scaling

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('L') #bgnet
        # return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        if filename.find('mat') > -1:
            data = scio.loadmat(filename)
            disp = data['LFT_disparity']
            where_are_nan = np.isnan(data['LFT_disparity'])
            # print("before:",where_are_nan)
            disp[where_are_nan] = -1
            where_are_nan = np.isnan(data['LFT_disparity'])
            # print("after:",where_are_nan)
            where_are_inf = np.isinf(data['LFT_disparity'])
            disp[where_are_inf] = -1
            gt_disp = disp.astype(np.float32)
            # data = np.array(data, dtype=np.float32) / 256.
        else:
            data = misc.imread(filename)
            # print(data)
            gt_disp = np.array(data, dtype=np.float32)
        return gt_disp

    def resize_image(self, image, width=0, height=0):
        shape = image.shape
        if len(shape)==2:
            image = image.expand(1, 1, shape[0], shape[1])
        else:
            image = image.expand(1, shape[0], shape[1], shape[2])
        # print("aaa:",image.shape)
        image = F.interpolate(image, size=[height, width], mode="bilinear")
        image = image.squeeze()
        return image

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath_rgb, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath_rgb, self.right_filenames[index]))
        if self.disp_filenames and self.datapath_d:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath_d, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            coff = int(self.scaling[1]) if self.scaling[0]=='x' else random.randint(2,int(self.scaling[1]))
            # if self.scaling[0]=='x':
            #     coff = int(self.scaling[1])
            # elif self.scaling[0]=='r':
            #     coff = random.randint(1,int(self.scaling[1]))
            # print("scaling coff = ",coff)
            input_w, input_h = 512, 256
            crop_w, crop_h = input_w*coff, input_h*coff

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # print(type(disparity))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform1() # bg:get_transform1
            left_img = processed(left_img)
            right_img = processed(right_img)

            left_img = self.resize_image(left_img, input_w, input_h)
            right_img = self.resize_image(right_img, input_w, input_h)
            disparity = torch.from_numpy(disparity)
            disparity = self.resize_image(disparity, input_w, input_h)/coff

            left_img_half = self.resize_image(left_img, input_w//2, input_h//2)
            right_img_half = self.resize_image(right_img, input_w//2, input_h//2)
            disparity_half = self.resize_image(disparity, input_w//2, input_h//2)/2

            # # bgnet
            left_img = torch.unsqueeze(left_img, 0)
            right_img = torch.unsqueeze(right_img, 0)
            left_img_half = torch.unsqueeze(left_img_half, 0)
            right_img_half = torch.unsqueeze(right_img_half, 0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_half": left_img_half,
                    "right_half": right_img_half,
                    "disparity_half": disparity_half,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]
                    }
        else:
            w, h = left_img.size
            resize_w, resize_h = 1024, 512
            coff = w/resize_w
            processed = get_transform1() # bg:get_transform1

            left_img = processed(left_img)
            right_img = processed(right_img)

            left_imgs = []
            right_imgs = []
            coffs = [0.25, 0.5, 0.75, 1]
            for i in range(len(coffs)):
                left_imgs.append(self.resize_image(left_img, (int)(resize_w*coffs[i]) // 1, (int)(resize_h*coffs[i]) // 1))
                right_imgs.append(self.resize_image(right_img, (int)(resize_w*coffs[i]) // 1, (int)(resize_h*coffs[i]) // 1))

            left_img_half = self.resize_image(left_img, resize_w // 2, resize_h // 2)
            right_img_half = self.resize_image(right_img, resize_w // 2, resize_h // 2)

            left_img = self.resize_image(left_img, resize_w, resize_h)
            right_img = self.resize_image(right_img, resize_w, resize_h)

            # # bgnet
            left_img = torch.unsqueeze(left_img, 0)
            right_img = torch.unsqueeze(right_img, 0)
            left_img_half = torch.unsqueeze(left_img_half, 0)
            right_img_half = torch.unsqueeze(right_img_half, 0)
            for i in range(len(coffs)):
                left_imgs[i] = torch.unsqueeze(left_imgs[i], 0)
                right_imgs[i] = torch.unsqueeze(right_imgs[i], 0)

            if disparity is None:
                return {"left": left_img,
                        "right": right_img,
                        "left_half": left_img_half,
                        "right_half": right_img_half,
                        "lefts": left_imgs,
                        "rights": right_imgs,
                        "width": w,
                        "height": h,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                disparity = torch.from_numpy(disparity)
                disparity_half = disparity
                # disparity = disparity.resize_(512, 1024)
                # disparity = disparity.expand(1, 1, left_img.size[1], left_img.size[2])
                # disparity = F.interpolate(disparity, size=[512, 1024], mode="bilinear")
                # disparity = disparity.squeeze()
                disparity_imgs = []
                for i in range(len(coffs)):
                    disparity_imgs.append(self.resize_image(disparity, (int)(resize_w * coffs[i]) // 1, (int)(resize_h * coffs[i]) // 1) * coffs[i] / coff)
                disparity_half = self.resize_image(disparity_half, resize_w // 2, resize_h // 2)/coff/2
                disparity = self.resize_image(disparity, resize_w, resize_h)/coff
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "left_half": left_img_half,
                        "right_half": right_img_half,
                        "disparity_half": disparity_half,
                        "lefts": left_imgs,
                        "rights": right_imgs,
                        "disparities": disparity_imgs,
                        "width": w,
                        "height": h,
                        "coff":coff,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
