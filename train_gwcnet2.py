from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import scipy
import scipy.misc
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import time
from tensorboardX import SummaryWriter
from Datasets import __datasets__
from Nets import __models__
from Utils import *
from Utils.utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings("ignore")
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--maxdisp_test', type=int, default=192, help='maximum disparity')
parser.add_argument('--test_water', type=bool, default=False, help='training batch size')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_rgb', required=True, help='data path')
parser.add_argument('--datapath_depth', required=True, help='data path')
parser.add_argument('--datapath_water_rgb', help='data path')
parser.add_argument('--datapath_water_depth', help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--water_testlist', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--loss_image', help='the directory to save loss image')
parser.add_argument('--result_path', help='result_path')
parser.add_argument('--result_path_water', help='result_path')
parser.add_argument('--logdir', help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--devices', help='the empty gpu to use')
parser.add_argument('--scaling', default="x1", help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# set GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath_rgb, args.datapath_depth, args.trainlist, True, args.scaling)
test_dataset = StereoDataset(args.datapath_rgb, args.datapath_depth, args.testlist, False, args.scaling)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)
if args.test_water:
    test_dataset_water = __datasets__["underwater"](args.datapath_water_rgb, args.datapath_water_depth,
                                                    args.water_testlist, False, args.scaling)
    TestImgLoader_water = DataLoader(test_dataset_water, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
if args.model=='bgnet' or args.model=='bgnet-plus':
    model = __models__[args.model]()
else:
    model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt,map_location='cuda:0')
    if args.loadckpt.endswith(".pth"):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name = 'module.'+key
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

len_plt = args.epochs
test_losses = np.zeros([len_plt])
D1s = np.zeros([len_plt])
EPEs = np.zeros([len_plt])
Thres3s = np.zeros([len_plt])


def train():
    # start_full_time = time.time()
    # for epoch_idx in range(start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
    #     total_train_loss = 0
    # 
    #     # training 180:训练
    #     for batch_idx, sample in enumerate(TrainImgLoader):
    #         global_step = len(TrainImgLoader) * epoch_idx + batch_idx
    #         start_time = time.time()
    #         do_summary = global_step % args.summary_freq == 0
    #         loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
    #         if do_summary:
    #             save_scalars(logger, 'train', scalar_outputs, global_step)
    #             save_images(logger, 'train', image_outputs, global_step)
    #         del scalar_outputs, image_outputs
    #         total_train_loss += loss
    #         print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
    #                                                                                    batch_idx,
    #                                                                                    len(TrainImgLoader), loss,
    #                                                                                    time.time() - start_time))
    #     print('Epoch {}/{}, total_train_loss = {:.3f}, full training time  = {:.3f} min'.format(epoch_idx, args.epochs,
    #                                                                                             total_train_loss, (
    #                                                                                                         time.time() - start_full_time) / 60))
    #     with open(args.result_path, 'a+') as f:
    #         f.writelines('Epoch {}/{}, total_train_loss = {:.3f}, full training time  = {:.3f} min\n'.format(epoch_idx,
    #                                                                                                          args.epochs,
    #                                                                                                          total_train_loss,
    #                                                                                                          (time.time() - start_full_time) / 60))
    #         # print(args.logdir)

        # saving checkpoints
        # if (epoch_idx + 1) % args.save_freq == 0:
        #     checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        # gc.collect()
        # # testing with kitti ： 14张测试
        # total_test_loss = 0
        # avg_test_scalars = AverageMeterDict()
        # for batch_idx, sample in enumerate(TestImgLoader):
        #     global_step = len(TestImgLoader) * epoch_idx + batch_idx
        #     start_time = time.time()
        #     do_summary = global_step % args.summary_freq == 0
        #     loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
        #     if do_summary:
        #         save_scalars(logger, 'test', scalar_outputs, global_step)
        #         save_images(logger, 'test', image_outputs, global_step)
        #     avg_test_scalars.update(scalar_outputs)
        #     del scalar_outputs, image_outputs
        #     total_test_loss += loss
        #     print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,batch_idx,
        #                                                                              len(TestImgLoader), loss,
        #                                                                              time.time() - start_time))
        #
        # print('Epoch {}/{}, total_test_loss = {:.3f}'.format(epoch_idx,args.epochs,total_test_loss))
        # with open(args.result_path, 'a+') as f:
        #         f.writelines('Epoch {}/{}, total_test_loss = {:.3f}\n'.format(epoch_idx,args.epochs,total_test_loss))
        # # print("train avg_test_scalars:", avg_test_scalars.dtype)
        # avg_test_scalars = avg_test_scalars.mean()
        # save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        # print("avg_test_scalars", avg_test_scalars)
        # with open(args.result_path, 'a+') as f:
        #         f.writelines("avg_test_scalars {}\n\n".format(avg_test_scalars))

        # testing with underwater ： 真实水下数据测试
        if args.test_water:
            # testing with water
            total_test_loss = 0
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader_water):
                global_step = len(TestImgLoader_water) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                # loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                loss, scalar_outputs, image_outputs = test_sample_water(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                total_test_loss += loss
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time))
            print('Epoch {}/{}, total_test_loss = {:.3f}'.format(epoch_idx, args.epochs, total_test_loss))
            with open(args.result_path_water, 'a+') as f:
                f.writelines(
                    'Epoch {}/{}, total_test_loss = {:.3f}\n'.format(epoch_idx, args.epochs, total_test_loss))
            # print("train avg_test_scalars:", avg_test_scalars.dtype)
            avg_test_scalars = avg_test_scalars.mean()
            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_test_scalars", avg_test_scalars)
            with open(args.result_path_water, 'a+') as f:
                f.writelines("avg_test_scalars {}\n\n".format(avg_test_scalars))

        gc.collect()


def val_supervise(concate=False):
    # testing
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the all checkpoint file
    for epoch_idx in range(len(all_saved_ckpts)):
        total_test_loss = 0
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[epoch_idx])
        print("loading the model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        # optimizer.load_state_dict(state_dict['optimizer'])
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader_water):
            global_step = batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample_water(sample, concate=concate, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)

            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            total_test_loss += loss
            print(' epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                   len(TestImgLoader_water), loss,
                                                                                   time.time() - start_time))
            # # 训练loss图
            # test_losses[epoch_idx] = total_test_loss
            # print(avg_test_scalars["EPE"][0])
            # EPEs[epoch_idx] = avg_test_scalars["EPE"][0]
            # D1s[epoch_idx] = avg_test_scalars["D1"][0]
            # Thres3s[epoch_idx] = avg_test_scalars["Thres3"][0]
            # fig1, ax1 = plt.subplots()
            # le = args.epochs
            # lns1 = ax1.plot(np.arange(le), test_losses, label="test_loss")
            # lns2 = ax1.plot(np.arange(le), EPEs, label="EPE")
            # lns3 = ax1.plot(np.arange(le), D1s, label="D1")
            # lns4 = ax1.plot(np.arange(le), Thres3s, label="Thres3")
            # ax1.set_xlabel('epoch')
            # ax1.set_ylabel('water test metric')
            # plt.legend(loc=0)
            # plt.savefig(args.loss_image)

        print('Epoch {}/{}, total_test_loss = {:.3f}'.format(epoch_idx, args.epochs, total_test_loss))
        with open(args.result_path, 'a+') as f:
            f.writelines('Epoch {}/{}, total_test_loss = {:.3f}\n'.format(epoch_idx, args.epochs, total_test_loss))
        # print("train avg_test_scalars:", avg_test_scalars.dtype)
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        with open(args.result_path, 'a+') as f:
            f.writelines("avg_test_scalars {}\n\n".format(avg_test_scalars))
        gc.collect()


def val_from_txt():
    # testing
    lines = read_all_lines(args.result_path)
    splits = [line.split() for line in lines]
    # total_test_loss = [x[-1] for x in splits]
    # 训练loss图
    test_losses = [float(x[-1]) for x in splits]
    fig1, ax1 = plt.subplots()
    le = args.epochs
    lns1 = ax1.plot(np.arange(le), test_losses, label="test_loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('validate loss')
    plt.legend(loc=0)
    plt.savefig(args.loss_image)
    print("Successfully painting loss png from txt!")


# train one sample
# def train_sample(sample, compute_metrics=False):
#     model.train()
# 
#     imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
#     imgL = imgL.cuda()
#     imgR = imgR.cuda()
#     disp_gt = disp_gt.cuda()
# 
# 
#     optimizer.zero_grad()
# 
#     # print(imgL.shape)
# 
#     try:
#         disp_ests = model(imgL, imgR)
#     except RuntimeError as exception:
#         if "out of memory" in str(exception):
#             print("WARNING: out of memory")
#             if hasattr(torch.cuda, 'empty_cache'):
#                 torch.cuda.empty_cache()
#         else:
#             raise exception
# 
#     # mask = disp_gt
#     mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
#     disp_gt = disp_gt.to(torch.float32)
#     loss = model_loss_gwc(disp_ests, disp_gt, mask)
# 
#     scalar_outputs = {"loss": loss}
# 
#     # disp_ests[0].shape: torch.Size([1, 256, 512])
#     # disp_gt.shape: torch.Size([1, 256, 512])
#     # imgL.shape: torch.Size([1, 3, 256, 512])
#     image_outputs = {"disp_ests": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
#     if compute_metrics:
#         with torch.no_grad():
#             # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
#             scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
#             scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
#             scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
#             scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
#             scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
#     loss.backward()
#     optimizer.step()
# 
#     return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, concate=False, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    if concate:
        imgL = torch.cat((imgL, imgL), dim=1)
        imgR = torch.cat((imgR, imgR), dim=1)
    disp_gt = disp_gt.cuda()
    # if args.dataset == "underwater":
        # width, height, coff = sample["width"], sample["height"], sample["coff"]
        # width = width.cuda()
        # height = height.cuda()
        # coff = coff.cuda().float()

    # print(imgL.shape)
    with torch.no_grad():
        # print("-----------------------------------",concate)
        disp_ests = model(imgL, imgR)
    disp_gt = disp_gt.to(torch.float32)

    mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)
    disp_ests = [
        F.interpolate(disp_est_half.expand(1, 1, disp_est_half.shape[1], disp_est_half.shape[2]), size=[512, 1024],
                      mode="bilinear").squeeze(1) for disp_est_half in disp_ests]
    loss = model_loss_gwc(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

# test one sample
@make_nograd_func
def test_sample_water(sample, concate=False, compute_metrics=True):
   model.eval()

   imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
   imgL_half, imgR_half, disp_gt_half = sample['left_half'], sample['right_half'], sample['disparity_half']
   imgL = imgL.cuda()
   imgR = imgR.cuda()
   imgL_half = imgL_half.cuda()
   imgR_half = imgR_half.cuda()
   if concate:
       imgL = torch.cat((imgL, imgL), dim=1)
       imgR = torch.cat((imgR, imgR), dim=1)
       imgL_half = torch.cat((imgL_half, imgL_half), dim=1)
       imgR_half = torch.cat((imgR_half, imgR_half), dim=1)
   disp_gt = disp_gt.cuda()
   # disp_gt_half = disp_gt_half.cuda()

   # print(imgL.shape)
   with torch.no_grad():
       # print("-----------------------------------",concate)
       disp_ests = model(imgL, imgR)
       disp_ests_half = model(imgL_half, imgR_half)
   disp_gt = disp_gt.to(torch.float32)
   # disp_gt_half = disp_gt_half.to(torch.float32)
   disp_ests_half = [F.interpolate(disp_est_half.expand(1, 1, disp_est_half.shape[1], disp_est_half.shape[2]), size=[512, 1024],mode="bilinear").squeeze(1) * 2 for disp_est_half in disp_ests_half]

   for i in range(len(disp_ests)):
       # print(i)
       left = 40
       right = 200

       mask_small = torch.where(((disp_gt > 0) & (disp_gt <= left)), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
       mask_mid = torch.where(((disp_gt > left) & (disp_gt <= right)), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
       mask_large = torch.where(((disp_gt > right) & (disp_gt < 384)), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
       
       disp_est_small = disp_ests_half[i] * mask_large
       
       similarity = torch.cosine_similarity(disp_ests_half[i], disp_ests[i])
       # print(similarity)
       weight2 = 0.5+0.5*similarity
       weight1 = 1 - weight2
       disp_est_mid = (disp_ests_half[i] * weight1 + disp_ests[i] * weight2) * mask_mid
       
       disp_est_large = disp_ests[i] * mask_small

       disp_ests[i] = disp_est_small + disp_est_mid + disp_est_large

       # similarity = torch.cosine_similarity(disp_ests_half[i], disp_ests[i])
       # # print(similarity)
       # weight1=0.5+0.5*similarity
       # weight2 = 1 - weight1
       # disp_ests[i] = disp_ests_half[i] * weight1 + disp_ests[i] * weight2

   # disp_ests = disp_ests_half
   mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)
   loss = model_loss_gwc(disp_ests, disp_gt, mask)

   scalar_outputs = {"loss": loss}
   image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

   scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
   scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
   scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
   scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
   scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

   # if compute_metrics:
       # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

   return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample_water_dra(sample, concate=False, compute_metrics=True):
   model.eval()

   imgsL, imgsR, disps_gt = sample['lefts'], sample['rights'], sample['disparities']
   n = len(imgsL)
   multi_res_ests = []
   for i in range(n):
       imgsL[i].cuda()
       imgsR[i].cuda()
       disps_gt[i].cuda()
       # print(imgsL[i].shape)

   # print(imgL.shape)
   with torch.no_grad():
       # print("-----------------------------------",concate)
       for i in range(n):
           # print(imgsL[i].shape[1])
           cur_ests = model(imgsL[i], imgsR[i])
           cur_ests = [F.interpolate(cur_est.expand(1, 1, cur_est.shape[1], cur_est.shape[2]), size=[512, 1024], mode="bilinear").squeeze(1) * 2 for cur_est in cur_ests]
           # print(cur_ests[0].shape)
           multi_res_ests.append(cur_ests)   
   

   # for i in range(len(disp_ests)):
       # left = 40
       # right = 200

       # masks = []
       # for i in range(n):
           # masks.append(torch.where(((disp_gt > 0) & (disp_gt <= left)), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0)))
           # disp_ests[i] = cur_ests[i] * mask[i]

       # # for i in range(n):
           # # disp_ests[i] = cur_ests[i] * mask[i]
       
       # similarity = torch.cosine_similarity(disp_ests_half[i], disp_ests[i])
       # # print(similarity)
       # weight2 = 0.5+0.5*similarity
       # weight1 = 1 - weight2
       # disp_est_mid = (disp_ests_half[i] * weight1 + disp_ests[i] * weight2) * mask_mid
       
       # disp_est_large = disp_ests[i] * mask_small
       
       # disp_ests[i] = disp_est_small + disp_est_mid + disp_est_large

       # # similarity = torch.cosine_similarity(disp_ests_half[i], disp_ests[i])
       # # # print(similarity)
       # # weight1=0.5+0.5*similarity
       # # weight2 = 1 - weight1
       # # disp_ests[i] = disp_ests_half[i] * weight1 + disp_ests[i] * weight2

   disp_ests = multi_res_ests[3]
   # disp_ests = disp_ests_half
   disp_gt = disps_gt[3].to(torch.float32)
   mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)
   loss = model_loss_gwc(disp_ests, disp_gt, mask)

   scalar_outputs = {"loss": loss}
   image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgsL": imgsL, "imgsR": imgsR}

   scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
   scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
   scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
   scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
   scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

   # if compute_metrics:
       # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

   return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    # val_from_txt()
    # concate = False
    # val_supervise(concate = concate)
    train()
