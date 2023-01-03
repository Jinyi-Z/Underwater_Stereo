from .discriminator import Discriminator_3D_1
from .discriminator import Discriminator_3D_2
from .discriminator import Discriminator_2D
from .disp_net import DispNet
from .resnet import resnet18
from .short_resnet import short_resnet9
from .gwc_net import GwcNet_GC
from .psm_net import PSMNet
from .gwc_net_concat import GwcNet_GC_Concat
import torch
import torch.optim as optim
from .psm_net import PSMNet as psm_net
from .bgnet import BGNet
from .bgnet_plus import BGNet_Plus

__models__ = {
    'resnet18': resnet18,
    'short_resnet9': short_resnet9,
    'Discriminator_3D_1': Discriminator_3D_1,
    'Discriminator_3D_2': Discriminator_3D_2,
    'Discriminator_2D': Discriminator_2D,
    "dispnet":DispNet,
    "psm_net":psm_net,
    "gwcnet-gc":GwcNet_GC,
    "gwcnet-gc-concat":GwcNet_GC_Concat,
    "bgnet":BGNet,
    "bgnet-plsu":BGNet_Plus
}