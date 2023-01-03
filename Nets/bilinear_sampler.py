import torch

def generate_image_left(right, disp_left):
    # print("right shape:",right.shape,disp_left.shape)
    warpped_lefts = []
    for i in range(len(disp_left)):
        warpped_lefts.append(bilinear_sampler_1d_h(right, -disp_left[i]))
    return warpped_lefts

def generate_image_right(left, disp_right):
    # print("left shape:",left.shape,disp_right.shape)
    return bilinear_sampler_1d_h(left, disp_right)

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
        x1 = torch.min(x1_f,  torch.tensor([_width_f - 1 + 2 * _edge_size]).cuda()).to(torch.int32)

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width)
        base_y0 = base + y0.long() * dim2
        idx_l = base_y0.cuda() + x0.long()
        idx_r = base_y0.cuda() + x1.long()

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
        # _height_f = 256.0
        # _width_f = 512.0
        # print("_height=",_height)
        # print("_width=",_width)
        y_t, x_t = torch.meshgrid((torch.linspace(0.0, _height_f - 1.0, _height),
                                  torch.linspace(0.0, _width_f - 1.0, _width)))

        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))

        x_t_flat = x_t_flat.repeat(_num_batch, 1)
        y_t_flat = y_t_flat.repeat(_num_batch, 1)

        x_t_flat = torch.reshape(x_t_flat, [-1])
        y_t_flat = torch.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat.cuda() + torch.reshape(x_offset, [-1]) * w_f

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
    # print(output.shape)
    return output