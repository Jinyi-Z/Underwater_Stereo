Underwater_Stereo
===================
Underwater Depth Estimation via Stereo Adaptation Networks
===================
This repo implements the training and testing of underwater stereo networks for "Underwater Depth Estimation via  Stereo Adaptation Networks" by Xinchen Ye, Yazhi Yuan, Jinyi Zhang, and et al. at DLUT.

Overview of the proposed underwater depth estimation pipeline via stereo adaptation networks
-------------------
<img width="884" alt="截屏2022-07-01 11 17 59" src="https://user-images.githubusercontent.com/78418629/176816760-004f7390-9cdb-4773-a399-c3510501ddfb.png">

Results
-------------------
<img width="924" alt="截屏2022-07-01 11 22 19" src="https://user-images.githubusercontent.com/78418629/176817219-e3e20824-f75f-4476-8524-f1d8db67b682.png">
Due to the lack of underwater training data, we first propose a depth-aware stereo image translation network to synthesize stylized underwater stereo images from terrestrial dataset, thus benefiting the effective training of depth estimation network. Then, considering the weak generalization to the real underwater data when only trained on the above synthetic data, we present a self-ensembling feature adaptation for depth estimation network to minimize the semantic domain discrepancy between synthetic and real underwater data. Meanwhile, we design a disparity rangead aptation module to address the problem of disparity rangemiss-match between both data, thus obtaining more accurate depth predictions for large-disparity-span underwater images.

Datasets
--------
KITTI: KITTI  is  a  real-world  dataset  with  terrestrialviews  which  provides  20000+  raw  stereo  image  data,  and 194/200 stereo images with sparse groundtruth disparities for KITTI2012/KITTI2015,  respectively. 

USD: USD collects 57 underwater stereo pairs from fourdifferent sites (Katzaa, Mikhmoret, Nachsholim and Satil) inIsrael with different characteristic attributes, where Satil isa shipwreck site (8 pairs) in the Red Sea (tropical water), and the other three are rocky reef environments, separated by Katzaa (15 pairs) in the Red Sea, Nachsholim (13 pairs) andMikhmoret (21 pairs) in the Mediterranean Sea (temperatewater). 

Environment
--------
The code supports Python 3.7

PyTorch (>= 1.7.0)

We uploaded the YML file of the environment. You can install the code environment of this paper through the following instructions:

`conda env create -f bgnet.yml`

Download pretrained models and datasets
--------
Download the pretrained model and datasets from the Baidu netdisk folder. Link:
https://pan.baidu.com/s/1mf_Kj0Ivme_tWBbj4OgeaA ，password : ftzo

Test
--------
`sh kitti_12_bg.sh`

Citation
--------
If you find this code useful, please cite:

`Xinchen Ye, Yazhi Yuan, Jinyi Zhang, et al., Underwater Depth Estimation via Unsupervised Stereo Adaptation Networks, submitted to TCSVT 2022.`
