import torch
import torch.nn.functional as F
from Utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor
import math

# Update D1 from >3px to >=3px & >5%
# matlab code:
# E = abs(D_gt - D_est);
# n_err = length(find(D_gt > 0 & E > tau(1) & E. / abs(D_gt) > tau(2)));
# n_total = length(find(D_gt > 0));
# d_err = n_err / n_total;

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        # print("len=",len(var.size()))
        # print(var.size(), vars[0].size())
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                print("masks[",idx,"].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[",idx,"].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def WP_metric(warpped_left, input_left, mask):
    warpped_left, input_left = warpped_left[mask], input_left[mask]
    return F.l1_loss(warpped_left, input_left, size_average=True)

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)

class metric_list(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, prediction, gt):
        for t in self.transforms:
            t(prediction, gt)
    def loss_get(self):
        results = []
        for t in self.transforms:
            acc = t.loss_get()
            info = {
                'metric': t.metric_name,
                'acc': acc
            }
            results.append(info)
        return results
    def reset(self):
        for t in self.transforms:
            t.reset()


# mean Square error
class RMS(object):
    def __init__(self, metric_name='RMS'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name
    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)
        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        diff = (up_prediction[:, :, 28:455, 24:585] - gt[:, :, 44:471, 40:601])
        square_diff = diff * diff
        rms_sum = float(square_diff.sum())
        b, c, h, w = square_diff.shape
        self.loss += rms_sum
        self.pixel_num += float(b*c*h*w)
    def loss_get(self, frac=4):
        return round(math.sqrt(self.loss / self.pixel_num), frac)
    def reset(self):
        self.loss = 0
        self.pixel_num = 0
        self.scaled_loss = 0
        self.scaled_pixel_num = 0

# average relative error
class REL(object):
    def __init__(self, metric_name='REL'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name
    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)
        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        abs_diff = (up_prediction[:, :, 28:455, 24:585] - gt[:, :, 44:471, 40:601]).abs()
        absrel_sum = float((abs_diff / gt[:, :, 44:471, 40:601]).sum())
        b, c, h, w = abs_diff.shape
        self.loss += absrel_sum
        self.pixel_num += float(b*c*h*w)
    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)
    def reset(self):
        self.loss = 0
        self.pixel_num = 0

# Mean log 10 error (log10)
class log10(object):
    def __init__(self, metric_name='log10'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name
    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)
        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        log10_diff = (torch.log10(up_prediction[:, :, 28:455, 24:585]) - torch.log10(gt[:, :, 44:471, 40:601])).abs()
        log10_sum = float(log10_diff.sum())
        b, c, h, w = log10_diff.shape
        self.loss += log10_sum
        self.pixel_num += float(b*c*h*w)
    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)
    def reset(self):
        self.loss = 0
        self.pixel_num = 0

# thresholded accuracy (deta)
class deta(object):
    def __init__(self, metric_name='deta', threshold=1.25):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name
        self.threshold = threshold
    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)
        up_prediction = F.upsample(prediction, [h_p * 4, w_p * 4], mode='bilinear', align_corners=True)
        up_prediction_region = up_prediction[:, :, 28:455, 24:585]
        gt_region = gt[:, :, 44:471, 40:601]
        deta_matrix = torch.cat((up_prediction_region / gt_region, gt_region/up_prediction_region), 1).max(1)[0]
        b, c, h, w = gt_region.shape
        self.loss += float((deta_matrix < self.threshold).sum())
        self.pixel_num += float(b * c * h * w)
    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)
    def reset(self):
        self.loss = 0
        self.pixel_num = 0