import cv2
import numpy as np
import torch
import os
from torchvision import utils as vutils
from torchvision import transforms
np.set_printoptions(threshold=np.inf)


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        rep = torch.unsqueeze(x, 1).repeat(1, n_repeats)
        return torch.reshape(rep, [-1])

    def _interpolate(im, x, y):
        # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            im = torch.nn.functional.pad(im, (0, 0, 1, 1, 1, 1), mode='constant')
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = torch.floor(x)
        y0_f = torch.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.to(torch.int32)
        y0 = y0_f.to(torch.int32)
        x1 = torch.min(x1_f,  torch.tensor([_width_f - 1 + 2 * _edge_size])).to(torch.int32)

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width)
        base_y0 = base + y0.long() * dim2
        idx_l = base_y0 + x0.long()
        idx_r = base_y0 + x1.long()

        im_flat = torch.reshape(im, [-1, _num_channels])

        pix_l = im_flat[idx_l]
        pix_r = im_flat[idx_r]

        weight_l = torch.unsqueeze(x1_f - x, 1)
        weight_r = torch.unsqueeze(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        w = 1
        w_f = float(w)
        y_t, x_t = torch.meshgrid(torch.linspace(0.0, _height_f - 1.0, _height),
                                  torch.linspace(0.0, _width_f - 1.0, _width))

        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))

        x_t_flat = x_t_flat.repeat(_num_batch, 1)
        y_t_flat = y_t_flat.repeat(_num_batch, 1)

        x_t_flat = torch.reshape(x_t_flat, [-1])
        y_t_flat = torch.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + torch.reshape(x_offset, [-1]) * w_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = torch.reshape(input_transformed, (_num_batch, _height, _width, _num_channels))
        return output

    input_images = input_images.permute(0, 2, 3, 1)
    _num_batch = input_images.shape[0]
    _height = input_images.shape[1]
    _width = input_images.shape[2]
    _num_channels = input_images.shape[3]
    # handle every item separately

    _height_f = float(_height)
    _width_f = float(_width)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)
    output = output.permute(0, 3, 1, 2)
    return output

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


#disp_path = './warp_error/disp_occ'
#left_path = './warp_error/GDWCT/colored_0_IR'
#right_path = './warp_error/GDWCT/colored_1_IR'
#error_path = './warp_error/GDWCT/error'
#/home/sda/yuanyazhi/dataset/KITTI-Katzaa0.8-4-10-30-0-true-warp34-temp123-10/2012/training/colored_0
k="KITTI-Katzaa0.8-4-10-30-0-true-warp34-temp123-10" # 0.30661538679557243
k_nw='KITTI-Katzaa0.8-4-30-0-true-1050' # 0.3124195153321325
m="KITTI-Michmoret0.8-4-10-30-0-true-warp34-temp123-13" # 0.2956585014701749
m_nw='KITTI-Michmoret0.8-4-30-0-true-13' # 0.3048648425323443
n="KITTI-Nachsholim0.8-4-10-30-0-true-warp34-temp123-13" # 0.28876237227709406
n_nw='KITTI-Nachsholim0.8-4-30-0-true-13' # 0.3007764665552355
s="KITTI-Satil0.8-4-10-30-0-true-warp34-temp123-8" # 0.3378867431869203
s_nw='KITTI-Satil0.8-4-30-0-true-8' # 0.3456569089524798

disp_path = './dataset/KITTI/2012/training/disp_occ'
left_path = './dataset/KITTI-Katzaa0.8-4-30-0-true-1050/2012/training/colored_0'
right_path = './dataset/KITTI-Katzaa0.8-4-30-0-true-1050/2012/training/colored_1'
error_path = './error_nowarp_k'
num_pix = 0
error = 0
index = 0
for file in os.listdir(disp_path):
    disp = cv2.imread(disp_path + '/' + file, flags=-1)
    disp = np.array(disp, dtype=np.float32) / 256.
    left = cv2.imread(left_path + '/' + file, flags=-1)
    right = cv2.imread(right_path + '/' + file, flags=-1)

    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    disp_tensor = transform(disp)
    left_tensor = transform(left)
    right_tensor = transform(right)
    left_tensor = torch.unsqueeze(left_tensor, 0)
    right_tensor = torch.unsqueeze(right_tensor, 0)

    warp_left_tensor = bilinear_sampler_1d_h(right_tensor, -disp_tensor)

    warp_left_tensor = torch.squeeze(warp_left_tensor, 0)
    left_tensor = torch.squeeze(left_tensor, 0)

    warp_left_numpy = warp_left_tensor[0].numpy()
    left_numpy = left_tensor[0].numpy()

    mask1 = np.where(disp > 0, np.full_like(disp, 1), np.full_like(disp, 0)).astype(np.uint8)
    mask2 = np.where(warp_left_numpy != 0, np.full_like(warp_left_numpy, 1), np.full_like(warp_left_numpy, 0)).astype(np.uint8)

    error_map = (left_numpy - warp_left_numpy) * 10

    error += np.sum(abs(error_map) * mask1 * mask2)
    num_pix += np.sum(mask1 * mask2)


    error_map = np.round(abs(error_map) * 256).astype(np.uint16)

    error_map = cv2.applyColorMap(cv2.convertScaleAbs(error_map, alpha=3), cv2.COLORMAP_JET)
    error_map[:,:,0] *= mask1
    error_map[:,:,1] *= mask1
    error_map[:,:,2] *= mask1
    error_map[:,:,0] *= mask2
    error_map[:,:,1] *= mask2
    error_map[:,:,2] *= mask2
    cv2.imwrite(error_path + '/' + file, error_map)

    index += 1
    print(str(index))

error /= num_pix
print(error)
