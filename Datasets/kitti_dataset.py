import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from Datasets.data_io import get_transform, get_transform1, read_all_lines


class KittiDataset(Dataset):
    def __init__(self, datapath_rgb, datapath_d, list_filename, training, scaling):
        self.datapath_rgb = datapath_rgb
        self.datapath_d = datapath_d
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            # print("left path[2]:",left_images[2],"-----right[2]:",right_images[2])
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('L') #bgnet

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

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
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform1()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # print("left/right filename:",self.left_filenames[index],self.right_filenames[index])

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform1() #bgnet
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
