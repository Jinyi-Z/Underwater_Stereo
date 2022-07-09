#!/usr/bin/env bash
set -x

Katzaa="KITTI-Katzaa0.8-4-10-30-0-true-warp34-temp123-10"
Michmoret="KITTI-Michmoret0.8-4-10-30-0-true-warp34-temp123-13"
Nachsholim="KITTI-Nachsholim0.8-4-10-30-0-true-warp34-temp123-13"
Satil="KITTI-Satil0.8-4-10-30-0-true-warp34-temp123-8"
#  Katzaa  Michmoret  Nachsholim  Satil
SCENE="Katzaa"

eval temp=$(echo \$$SCENE)
WATER_NAME=$temp
DATAPATH_RGB="/dataset/"$WATER_NAME"/2012/"
DATAPATH_DEPTH="/dataset/KITTI/2012/"
DATAPATH_WATER="/dataset/UnderwaterStereoDataset/"
#LOAD_TRAIN="sf15-kt12"
# /home/sba/caoxiao/code/StereoNet_pytorch/Checkpoints/kitti12-fake/bests/K-xie/checkpoint_000211.ckpt
# $SERVER/code/StereoNet_pytorch/Checkpoints/scene-flow/gwcnet-gc/checkpoint_000015.ckpt \

#	--loadckpt $SERVER/code/StereoNet_pytorch/Checkpoints/bgnet/kitti_12_BGNet.pth  \

python $SERVER/code/StereoNet_pytorch/train_gwcnet2.py \
	--dataset kitti  --test_water True \
	--datapath_rgb $SERVER$DATAPATH_RGB  --datapath_depth $SERVER$DATAPATH_DEPTH \
	--datapath_water_rgb $SERVER$DATAPATH_WATER  --datapath_water_depth $SERVER$DATAPATH_WATER \
	--model bgnet --epochs 300 --devices 3 --lrepochs "200:10" --maxdisp_test 384 \
	--trainlist $SERVER/code/StereoNet_pytorch/Filenames/kitti/kitti12_train.txt \
	--testlist $SERVER/code/StereoNet_pytorch/Filenames/kitti/kitti12_val.txt \
	--water_testlist $SERVER/code/StereoNet_pytorch/Filenames/underwater/${SCENE}_test.txt \
	--loadckpt $SERVER/code/StereoNet_pytorch/Checkpoints/bgnet/kitti_12_BGNet.pth  \
	--result_path $SERVER/code/StereoNet_pytorch/Results/bgnet/kitti12-fake/$SCENE.txt\
	--result_path_water $SERVER/code/StereoNet_pytorch/Results/bgnet/kitti12-fake/$SCENE.txt\
	--logdir $SERVER/code/StereoNet_pytorch/Checkpoints/bgnet-fake/$SCENE \
    --batch_size 6 --test_batch_size 1

