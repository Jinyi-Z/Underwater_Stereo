Underwater_Stereo
===================
Underwater Depth Estimation via Unsupervised Stereo Adaptation Networks
===================
This repo implements the training and testing of underwater stereo networks for "Underwater Depth Estimation via Unsupervised Stereo Adaptation Networks" by Xinchen Ye, Yazhi Yuan, Jinyi Zhang, and et al. at DLUT.

Overview of the proposed underwater depth estimation pipeline via unsupervised stereo adaptation networks
-------------------
<img width="884" alt="截屏2022-07-01 11 17 59" src="https://user-images.githubusercontent.com/78418629/176816760-004f7390-9cdb-4773-a399-c3510501ddfb.png">

Results
-------------------
<img width="924" alt="截屏2022-07-01 11 22 19" src="https://user-images.githubusercontent.com/78418629/176817219-e3e20824-f75f-4476-8524-f1d8db67b682.png">
<img width="848" alt="截屏2022-07-01 11 22 43" src="https://user-images.githubusercontent.com/78418629/176817241-5837625b-0f71-49f8-8240-08d7d2c4ff41.png">
Due  to  the  lack  of  underwatertraining   data,   we   first   propose   a   depth-aware   stereo   imagetranslation   network   to   synthesize   stylized   underwater   stereoimages   from   terrestrial   dataset,   thus   benefiting   the   effectivetraining  of  depth  estimation  network.  Then,  considering  theweak  generalization  to  the  real  underwater  data  when  onlytrained on the above synthetic data, we present a self-ensemblingfeature  adaptation  for  depth  estimation  network  to  minimizethe   semantic   domain   discrepancy   between   synthetic   and   re-al  underwater  data.  Meanwhile,  we  design  a  disparity  rangeadaptation  module  to  address  the  problem  of  disparity  rangemiss-match  between  both  data,  thus  obtaining  more  accuratedepth  predictions  for  large-disparity-span  underwater  images.

Datasets
--------
KITTI: KITTI  is  a  real-world  dataset  with  terrestrialviews  which  provides  20000+  raw  stereo  image  data,  and 194/200 stereo images with sparse groundtruth disparities for KITTI2012/KITTI2015,  respectively. 

USD: USD collects 57 underwater stereo pairs from fourdifferent sites (Katzaa, Mikhmoret, Nachsholim and Satil) inIsrael with different characteristic attributes, where Satil isa shipwreck site (8 pairs) in the Red Sea (tropical water), and the other three are rocky reef environments, separated by Katzaa (15 pairs) in the Red Sea, Nachsholim (13 pairs) andMikhmoret (21 pairs) in the Mediterranean Sea (temperatewater). 

Environment
--------
We uploaded the YML file of the environment. You can install the code environment of this paper through the following instructions:

`conda env create -f bgnet.yml`

Download pretrained models
--------
Download the pretrained model from the Baidu netdisk folder. Link:
