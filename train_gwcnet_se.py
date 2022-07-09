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
import time
from tensorboardX import SummaryWriter
from Datasets import __datasets__
from Nets import __models__
from Nets.warp import disp_warp
from Utils import *
from Utils.utils import *
# from Utils.warp import *
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
parser.add_argument('--datapath_water', help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--water_testlist', help='testing list')
parser.add_argument('--water_trainlist', type=str, help='training list for target samples')
parser.add_argument('--ini_teacher', action='store_true', help='initiate the weights of teacher')

# parser.add_argument('--trainlist', required=True, help='training list')
# parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--loss_image', help='the directory to save loss image')
parser.add_argument('--result_path_student', help='result_path')
parser.add_argument('--result_path_water', help='result_path')
parser.add_argument('--logdir', help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--devices', help='the empty gpu to use')
parser.add_argument('--scaling', default="x1", help='the frequency of saving checkpoint')
parser.add_argument('--scene', type=str, help='which scene of target data')

parser.add_argument('--use_self_ensembling', action='store_true', help='use self-ensembling to adapt target domain during training')
parser.add_argument('--teacher_alpha', type=float, default=0.99, help='teacher alpha in EMA.')
parser.add_argument('--st_weight_max', type=float, default=1.0, help='self-ensembling weight.')
parser.add_argument('--dra', action='store_true', help='use disparity range adaptation')

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
if args.dataset == 'se':
    train_dataset = StereoDataset(args.datapath_rgb, args.datapath_depth, args.trainlist, args.datapath_water, args.water_trainlist, True, args.scaling, args.scene)
    test_dataset = StereoDataset(args.datapath_water, args.datapath_water, args.water_trainlist, args.water_testlist, args.water_testlist, False, args.scaling, args.scene)
else:
    train_dataset = StereoDataset(args.datapath_rgb,args.datapath_depth, args.trainlist, True, args.scaling)
    test_dataset = StereoDataset(args.datapath_rgb,args.datapath_depth,  args.testlist, False, args.scaling)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)
if args.test_water:
    test_dataset_water = __datasets__["underwater"](args.datapath_water, args.datapath_water, args.water_testlist, False, args.scaling)
    TestImgLoader_water = DataLoader(test_dataset_water, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
if args.model=='bgnet' or args.model=='bgnet-plus':
    model_student = __models__[args.model]()
else:
    model_student = __models__[args.model](args.maxdisp)

model_student = nn.DataParallel(model_student)
model_student.cuda()
optimizer_student = optim.Adam(model_student.parameters(), lr=args.lr, betas=(0.9, 0.999))

if args.use_self_ensembling:
    if args.model == 'bgnet' or args.model == 'bgnet-plus':
        model_teacher = __models__[args.model]()
    else:
        model_teacher = __models__[args.model](args.maxdisp)
    for name, param in model_teacher.named_parameters():
        param.requires_grad = False
    model_teacher = nn.DataParallel(model_teacher)
    model_teacher.cuda()

    student_params = list(model_student.parameters())
    teacher_params = list(model_teacher.parameters())
    optimizer_teacher = WeightEMA(teacher_params, student_params, alpha=args.teacher_alpha)


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
    model_student.load_state_dict(state_dict['model'])
    optimizer_student.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    pretrained_dict = torch.load(args.loadckpt,map_location='cuda:0')
    model_student.load_state_dict(pretrained_dict['model'])
    if args.ini_teacher:
        # teacher_dict = model_teacher.state_dict()
        # teacher_dict.update(pretrained_dict['model'])
        # model_teacher.load_state_dict(teacher_dict)
        model_teacher.load_state_dict(pretrained_dict['model'])

print("start at epoch {}".format(start_epoch))

len_plt = args.epochs
test_losses = np.zeros([len_plt])
# D1s = np.zeros([len_plt])
# EPEs = np.zeros([len_plt])
# Thres3s = np.zeros([len_plt])

def train():
    start_full_time = time.time()
    for epoch_idx in range(start_epoch, args.epochs):

        adjust_learning_rate(optimizer_student, epoch_idx, args.lr, args.lrepochs)
        total_train_loss = 0

        # training 180:训练
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            st_weight = args.st_weight_max * sigmoid_rampup(epoch_idx, args.epochs)
            if args.use_self_ensembling:
                loss, scalar_outputs, image_outputs = train_sample_se(sample, st_weight, compute_metrics=do_summary)
                loss_sup = scalar_outputs['loss']
                loss_con = scalar_outputs['lcon']
                print('Epoch {}/{}, Iter {}/{}, loss_sup = {:.3f}, loss_con = {:.3f}, time = {:.3f}'.format(epoch_idx,args.epochs,batch_idx,len(TrainImgLoader),loss_sup, loss_con,time.time() - start_time))
            else:
                loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
                print('Epoch {}/{}, Iter {}/{}, loss = {:.3f}, time = {:.3f}'.format(epoch_idx,args.epochs,batch_idx, len(TrainImgLoader), loss, time.time() - start_time))
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            total_train_loss += loss
        
        print('Epoch {}/{}, total_train_loss = {:.3f}, full training time  = {:.3f} min'.format(epoch_idx, args.epochs,total_train_loss,(time.time()-start_full_time)/60))
        #with open(args.result_path_water, 'a+') as f:
         #   f.writelines('Epoch {}/{}, total_train_loss = {:.3f}, full training time  = {:.3f} min\n'.format(epoch_idx, args.epochs,total_train_loss,(time.time()-start_full_time)/60))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data_student = {'epoch': epoch_idx, 'model': model_student.state_dict(), 'optimizer': optimizer_student.state_dict()}
            torch.save(checkpoint_data_student, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

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
                loss, scalar_outputs, image_outputs = test_sample_water_dra(sample, compute_metrics=do_summary)
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

def val_supervise(concate = False):
    # testing
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the all checkpoint file
    for epoch_idx in range(len(all_saved_ckpts)):
        # testing with water
        total_test_loss = 0
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader_water):
            global_step = len(TestImgLoader_water) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            # loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            loss, scalar_outputs, image_outputs = test_sample_water_dra(sample, compute_metrics=do_summary)
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
        # print("avg_test_scalars", avg_test_scalars)
        with open(args.result_path_water, 'a+') as f:
            f.writelines("avg_test_scalars {}\n\n".format(avg_test_scalars))
        gc.collect()

def val_from_txt():
    # testing
    lines = read_all_lines(args.result_path_water)
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

# train one sample using self-ensembling
def train_sample_se(sample, st_weight, compute_metrics=False):
    model_student.train()

    imgL, imgR, disp_gt, targetL1, targetR1, targetL2, targetR2 = sample['left'], sample['right'], sample['disparity'], sample['left_target_1'], sample['right_target_1'], sample['left_target_2'], sample['right_target_2']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    # print("shape imgL:",imgL.shape)
    disp_gt = disp_gt.cuda()
    # targetL = torch.cat((targetL1, targetL2), 0)
    # targetR = torch.cat((targetR1, targetR2), 0)
    targetL = targetL1.cuda()
    targetR = targetR1.cuda()
    #print("shape targetL:",targetL.shape)

    optimizer_student.zero_grad()

    try:
        disp_ests = model_student(imgL, imgR)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss_gwc(disp_ests, disp_gt, mask)

    noise = torch.clamp(torch.randn_like(targetL) * 0.1, -0.002, 0.002)

    student_ests = model_student(targetL, targetR)
    teacher_ests = model_teacher(targetL + noise, targetR + noise)
    loss_consistency = aug_loss(student_ests, teacher_ests) * st_weight
    #print(loss_consistency)
    #print(loss)
    # warpped_lefts = generate_image_left(imgR, disp_ests)
    # # warpped_left = warpped_left.to(torch.float32)
    # loss = model_loss_gwc_warp(warpped_lefts, imgL)

    scalar_outputs = {"loss": loss, "lcon": loss_consistency}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss_total = loss + loss_consistency
    loss_total.backward()
    optimizer_student.step()
    if args.use_self_ensembling:
        optimizer_teacher.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# train one sample
def train_sample(sample, compute_metrics=False):
    model_student.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # left_name, right_name = sample['left_filename'], sample['right_filename']
    # print("train image: ",left_name,"---", right_name)

    # print(imgL.shape)

    optimizer_student.zero_grad()

    # print(imgL.shape)
    try:
        noise0 = torch.clamp(torch.randn_like(imgL) * 0.1, -0.002, 0.002)
        noise1 = torch.clamp(torch.randn_like(imgL) * 0.1, -0.002, 0.002)
        # disp_ests = model_student(imgL, imgR)
        disp_ests = model_student(imgL + noise0, imgR + noise1)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception

    # mask = disp_gt
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_gt = disp_gt.to(torch.float32)
    loss = model_loss_gwc(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}

    # disp_ests[0].shape: torch.Size([1, 256, 512])
    # disp_gt.shape: torch.Size([1, 256, 512])
    # imgL.shape: torch.Size([1, 3, 256, 512])
    image_outputs = {"disp_ests": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer_student.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

# test one sample
@make_nograd_func
def test_sample(sample, concate=False, compute_metrics=True):
    model_student.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    if concate:
        imgL = torch.cat((imgL, imgL), dim=1)
        imgR = torch.cat((imgR, imgR), dim=1)
    disp_gt = disp_gt.cuda()
    # if args.dataset == "underwater":
    #     width, height, coff = sample["width"], sample["height"], sample["coff"]
    #     width = width.cuda()
    #     height = height.cuda()
    #     coff = coff.cuda().float()

    # print(imgL.shape)
    with torch.no_grad():
        # print("-----------------------------------",concate)
        disp_ests = model_student(imgL, imgR)
    disp_gt = disp_gt.to(torch.float32)

    mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)
    disp_ests = [F.interpolate(disp_est_half.expand(1, 1, disp_est_half.shape[1], disp_est_half.shape[2]), size=[512, 1024], mode="bilinear").squeeze(1) for disp_est_half in disp_ests]
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
def test_sample_water1(sample, concate=False, compute_metrics=True):
    model_student.eval()

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
        disp_ests = model_student(imgL, imgR)
        disp_ests_half = model_student(imgL_half, imgR_half)
    disp_gt = disp_gt.to(torch.float32)
    # disp_gt_half = disp_gt_half.to(torch.float32)

    disp_ests_half = [F.interpolate(disp_est_half.expand(1, 1, disp_est_half.shape[1], disp_est_half.shape[2]), size=[512, 1024], mode="bilinear").squeeze(1)*2 for disp_est_half in disp_ests_half]

    for i in range(len(disp_ests)):
        mask_small = torch.where(((disp_gt > 0) & (disp_gt <= 50)), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
        mask_mid = torch.where((disp_gt > 100) & (disp_gt <= 250), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
        mask_large = torch.where((disp_gt > 50) & (disp_gt < 384), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))
        disp_est_small = disp_ests_half[i]*mask_large
        disp_est_large = disp_ests[i]*mask_small
        disp_ests[i] = disp_est_small + disp_est_large
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
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample_water(sample, concate=False, compute_metrics=True):
    model_student.eval()

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
        disp_ests = model_student(imgL, imgR)
        disp_ests_half = model_student(imgL_half, imgR_half)
    disp_gt = disp_gt.to(torch.float32)
    # disp_gt_half = disp_gt_half.to(torch.float32)

    disp_ests_half = [F.interpolate(disp_est_half.expand(1, 1, disp_est_half.shape[1], disp_est_half.shape[2]), size=[512, 1024], mode="bilinear").squeeze(1)*2 for disp_est_half in disp_ests_half]

    for i in range(len(disp_ests)):
        left = 40
        right = 200

        mask_small = torch.where(((disp_gt > 0) & (disp_gt <= left)), torch.full_like(disp_gt, 1),
                                 torch.full_like(disp_gt, 0))
        mask_mid = torch.where(((disp_gt > left) & (disp_gt <= right)), torch.full_like(disp_gt, 1),
                               torch.full_like(disp_gt, 0))
        mask_large = torch.where(((disp_gt > right) & (disp_gt < 384)), torch.full_like(disp_gt, 1),
                                 torch.full_like(disp_gt, 0))

        disp_est_small = disp_ests_half[i] * mask_large

        similarity = torch.cosine_similarity(disp_ests_half[i], disp_ests[i])
        # print(similarity)
        weight2 = 0.5 + 0.5 * similarity
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
    model_student.eval()
    filename = sample['left_filename']
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgsL, imgsR, disps_gt = sample['lefts'], sample['rights'], sample['disparities']

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    n = len(imgsL)
    disp_ests = []
    for i in range(n):
        imgsL[i].cuda()
        imgsR[i].cuda()
        disps_gt[i].cuda()

    disp_gt = disp_gt.cuda()
    disp_gt = disp_gt.to(torch.float32)
    mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)

    with torch.no_grad():
        scaling = [4, 2, 4/3, 1]
        for i in range(n):
            cur_ests = model_student(imgsL[i], imgsR[i])
            cur_ests = [F.interpolate(cur_est.expand(1, 1, cur_est.shape[1], cur_est.shape[2]), size=[512, 1024],
                                      mode="bilinear").squeeze(1) * scaling[i] for cur_est in cur_ests]
            disp_ests.append(cur_ests[-1])

    if args.dra:
        for i in range(1, len(disp_ests)):
            
            a = 2    // 1-4
            b = 0.5  // 0-1
            D_scale = torch.abs(disp_ests[i]-disp_ests[i-1])
            U_scale = torch.where((D_scale > 192/a), torch.full_like(disp_gt, 1), torch.full_like(disp_gt, 0))

            warp_left = disp_warp(imgR, disp_ests[i])
            D_warp = torch.abs(imgL - warp_left)
            U_warp = 1-torch.exp(-b*D_warp)
            # error_tensor = U_warp.numpy()
            M_i = torch.mul(U_scale, U_warp)
            disp_ests[i] = M_i * disp_ests[i-1] + (1-M_i) * disp_ests[i]
            disp_ests[i] = torch.squeeze(disp_ests[i],0)

            # left = 40
            # right = 200

            # mask_small = torch.where(((disp_gt > 0) & (disp_gt <= left)), torch.full_like(disp_gt, 1),
                                     # torch.full_like(disp_gt, 0))
            # mask_mid = torch.where(((disp_gt > left) & (disp_gt <= right)), torch.full_like(disp_gt, 1),
                                   # torch.full_like(disp_gt, 0))
            # mask_large = torch.where(((disp_gt > right) & (disp_gt < 384)), torch.full_like(disp_gt, 1),
                                     # torch.full_like(disp_gt, 0))

            # disp_est_small = disp_ests[i-1] * mask_large
            # # print(disp_ests[i-1].shape) #torch.Size([1, 512, 1024])
            # similarity = torch.cosine_similarity(disp_ests[i-1], disp_ests[i])
            # # similarity = 0.5 + 0.5 * similarity
            # similarity = torch.sigmoid_(similarity)
            # # print(i,":","min=",torch.min(similarity), "max=",torch.max(similarity))


            # print(error_tensor)
            # if not os.path.exists(os.path.join("/data3T_1/yuanyazhi/code/StereoNet_pytorch/warp_imgs/" + filename[0].split('/')[-2])):
            #     os.makedirs(os.path.join("/data3T_1/yuanyazhi/code/StereoNet_pytorch/warp_imgs/" + filename[0].split('/')[-2]))
            # fn = os.path.join("/data3T_1/yuanyazhi/code/StereoNet_pytorch/warp_imgs/" + filename[0].split('/')[-2] + '/warp_error' + (str)(i) +'.png')
            # plt.imsave(fn, np.round(error_tensor * 256).astype(np.uint16),cmap='plasma')
            # warp_left_tensor = disp_warp(imgR, disp_gt)
            # left_tensor = torch.unsqueeze(imgL, 0)
            # right_tensor = torch.unsqueeze(imgR, 0)
            #
            # warp_left_tensor = torch.squeeze(warp_left_tensor, 0)
            # left_tensor = torch.squeeze(left_tensor, 0)
            #
            # warp_left_numpy = warp_left_tensor.numpy()
            # left_numpy = left_tensor.numpy()
            #
            # # mask = (disp_gt < args.maxdisp_test) & (disp_gt > 0)
            # mask1 = np.where(disp_gt > 0, np.full_like(disp_gt, 1), np.full_like(disp_gt, 0)).astype(np.uint8)
            # mask2 = np.where(warp_left_numpy != 0, np.full_like(warp_left_numpy, 1),
            #                  np.full_like(warp_left_numpy, 0)).astype(np.uint8)
            # error_map = left_numpy - warp_left_numpy
            # error = np.sum(abs(error_map) * mask1 * mask2)
            # num_pix = np.sum(mask1 * mask2)
            # error /= num_pix
            #
            # u_warp2 = warp_2(disp_ests[i], right, mask, args)
            # u_warp1 = warp_2(disp_ests[i-1], right, mask, args)
            # u2 = u_sim2 + u_warp2
            # u1 = u_sim1 + u_warp1
            # print("1:",u1,"=",u_sim1,"+",u_warp1)
            # print("2:",u2,"=",u_sim2,"+",u_warp2)
            # disp_est_mid = (disp_ests[i-1] * (1-similarity) + disp_ests[i] * similarity) * mask_mid
            # disp_est_large = disp_ests[i] * mask_small

            # disp_ests[i] = disp_est_small + disp_est_mid + disp_est_large

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


if __name__ == '__main__':
    # val_from_txt()
    # concate = False
    # val_supervise(concate = concate)
    train()
