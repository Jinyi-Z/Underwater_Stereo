#!/usr/bin/env bash
set -x
#SERVER="/home/sba/yuanyazhi"
SERVER="/data3T_1/yuanyazhi"
#SERVER="/home/sba/caoxiao"
#SERVER="/data2T_1/yuanyazhi"
#SERVER="/data4T_1/yuanyazhi"

Katzaa="KITTI-Katzaa0.8-4-10-30-0-true-warp34-temp123-10" # ok
Michmoret="KITTI-Michmoret0.8-4-10-30-0-true-warp34-temp123-13"
Nachsholim="KITTI-Nachsholim0.8-4-10-30-0-true-warp34-temp123-13"
Satil="KITTI-Satil0.8-4-10-30-0-true-warp34-temp123-8"
SCENE="Michmoret"

eval temp=$(echo \$$SCENE)
WATER_NAME=$temp
SOURCE_RGB="/dataset/"$WATER_NAME"/2012/"
SOURCE_DEPTH="/dataset/KITTI/2012/"
TARGET_PATH="/dataset/UnderwaterStereoDataset/"
SCALING="x3"

#/home/sda/yuanyazhi/code/StereoNet_pytorch/Checkpoints/scene-flow/gwcnet-gc/checkpoint_000015.ckpt \

LOAD_Katzaa="/data3T_1/yuanyazhi/code/StereoNet_pytorch/Checkpoints/bgnet-fake/Katzaa/checkpoint_000098.ckpt"
LOAD_Michmoret="/data3T_1/yuanyazhi/code/StereoNet_pytorch/Checkpoints/bgnet-fake/Michmoret/checkpoint_000144.ckpt"
LOAD_Nachsholim="/data3T_1/yuanyazhi/code/StereoNet_pytorch/Checkpoints/bgnet-fake/Nachsholim/checkpoint_000053.ckpt"
LOAD_Satil="/data3T_1/yuanyazhi/code/StereoNet_pytorch/Checkpoints/bgnet-fake/Satil/checkpoint_000098.ckpt"
temp1=LOAD_$SCENE
eval temp2=$(echo \$$temp1)
LOAD=$temp2

python $SERVER/code/StereoNet_pytorch/train_gwcnet_se.py \
	--dataset se --devices 3 --scaling $SCALING  --test_water True \
	--use_self_ensembling --ini_teacher --lr 0.001 --teacher_alpha 0.99 --st_weight_max 0.001 --maxdisp 192 --maxdisp_test 384 \
	--model bgnet --epochs 300 --lrepochs "200:10" --scene $SCENE \
	--datapath_rgb $SERVER$SOURCE_RGB --datapath_depth $SERVER$SOURCE_DEPTH \
	--datapath_water $SERVER$TARGET_PATH \
	--loadckpt $LOAD \
	--trainlist $SERVER/code/StereoNet_pytorch/Filenames/kitti/kitti12_train.txt \
	--testlist $SERVER/code/StereoNet_pytorch/Filenames/underwater/${SCENE}_test.txt  \
	--water_trainlist $SERVER/code/StereoNet_pytorch/Filenames/underwater/${SCENE}_all.txt \
	--water_testlist $SERVER/code/StereoNet_pytorch/Filenames/underwater/${SCENE}_test.txt \
	--result_path_water $SERVER/code/StereoNet_pytorch/Results/bgnet/kitti12-se/$SCENE-6.txt\
	--logdir $SERVER/code/StereoNet_pytorch/Checkpoints/bgnet-se/$SCENE-6 \
     --batch_size 6 --test_batch_size 1
