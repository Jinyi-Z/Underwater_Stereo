import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from Datasets.data_io import get_transform, get_transform1, read_all_lines
import torch
import scipy.io as scio
import torch.nn.functional as F


class SelfEnsemblingDataset(Dataset):
    def __init__(self, datapath, datapath_depth, trainlist, target_path, target_list, training, scaling, scene):
        self.datapath = datapath
        self.datapath_depth = datapath_depth
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(trainlist)
        self.target_path = target_path
        self.left_target_filenames, self.right_target_filenames, self.disp_target_filenames = self.load_path(target_list)
        self.training = training
        self.scaling = scaling
        if self.training:
            assert self.disp_filenames is not None
        scene_num = {"Katzaa": 13, "Michmoret": 18, "Nachsholim": 11, "Satil": 7}
        self.scene_num = scene_num[scene]

    def load_path(self, trainlist):
        lines = read_all_lines(trainlist)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('L') #bgnet

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_disp_target(self, filename):
        data = scio.loadmat(filename)
        data = np.array(data['LFT_disparity'], dtype=np.float32)
        data = np.where(np.isnan(data), np.full_like(data, -1), data)
        data = np.where(np.isinf(data), np.full_like(data, -1), data)
        return data

    def resize_image(self, image, width=0, height=0):
        shape = image.shape
        if len(shape)==2:
            image = image.expand(1, 1, shape[0], shape[1])
        else:
            image = image.expand(1, shape[0], shape[1], shape[2])
        image = F.interpolate(image, size=[height, width], mode="bilinear")
        image = image.squeeze()
        return image

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        left_target = self.load_image(os.path.join(self.target_path, self.left_target_filenames[index % self.scene_num]))
        right_target = self.load_image(os.path.join(self.target_path, self.right_target_filenames[index % self.scene_num]))

        if self.disp_filenames:  # has disparity ground truth
            if self.training:
                disparity = self.load_disp(os.path.join(self.datapath_depth, self.disp_filenames[index]))
            else:
                # disparity = self.load_disp_target(os.path.join(self.datapath, self.disp_target_filenames[index % self.scene_num]))
                disparity = self.load_disp_target(os.path.join(self.target_path, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            # source sample
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform1() #bgnet
            left_img = processed(left_img)
            right_img = processed(right_img)

            # target sample
            w_target, h_target = left_target.size

            coff = int(self.scaling[1]) if self.scaling[0] == 'x' else random.randint(2, int(self.scaling[1]))
            input_w, input_h = 512, 256 # 480, 224
            crop_w_target, crop_h_target = input_w * coff, input_h * coff

            x1_target = random.randint(0, w_target - crop_w_target)
            y1_target = random.randint(0, h_target - crop_h_target)

            x2_target = random.randint(0, w_target - crop_w_target)
            y2_target = random.randint(0, h_target - crop_h_target)

            # random crop
            left_target_1 = left_target.crop((x1_target, y1_target, x1_target + crop_w_target, y1_target + crop_h_target))
            right_target_1 = right_target.crop((x1_target, y1_target, x1_target + crop_w_target, y1_target + crop_h_target))

            left_target_2 = left_target.crop((x2_target, y2_target, x2_target + crop_w_target, y2_target + crop_h_target))
            right_target_2 = right_target.crop((x2_target, y2_target, x2_target + crop_w_target, y2_target + crop_h_target))

            # to tensor, normalize
            processed = get_transform1() #bgnet
            left_target_1 = processed(left_target_1)
            right_target_1 = processed(right_target_1)
            left_target_2 = processed(left_target_2)
            right_target_2 = processed(right_target_2)

            left_target_1 = self.resize_image(left_target_1, input_w, input_h)
            right_target_1 = self.resize_image(right_target_1, input_w, input_h)
            left_target_2 = self.resize_image(left_target_2, input_w, input_h)
            right_target_2 = self.resize_image(right_target_2, input_w, input_h)

            # # bgnet
            # print("before:",left_img.shape)
            left_target_1 = torch.unsqueeze(left_target_1, 0)
            right_target_1 = torch.unsqueeze(right_target_1, 0)
            left_target_2 = torch.unsqueeze(left_target_2, 0)
            right_target_2 = torch.unsqueeze(right_target_2, 0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_target_1": left_target_1,
                    "right_target_1": right_target_1,
                    "left_target_2": left_target_2,
                    "right_target_2": right_target_2}
        else:
            # w, h = left_img.size
            #
            # # normalize
            # processed = get_transform()
            # left_img = processed(left_img).numpy()
            # right_img = processed(right_img).numpy()
            #
            # top_pad = right_pad = 0

            w, h = left_img.size
            resize_w, resize_h = 1024, 512
            coff = w/resize_w
            processed = get_transform1() #bgnet

            left_img = processed(left_img)
            right_img = processed(right_img)

            left_img = self.resize_image(left_img, resize_w, resize_h)
            right_img = self.resize_image(right_img, resize_w, resize_h)

            # # pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            #
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            #
            # # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                disparity = torch.from_numpy(disparity)
                disparity = self.resize_image(disparity, resize_w, resize_h) / coff
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "width": w,
                        "height": h,
                        "coff": coff,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "width": w,
                        "height": h,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
