import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

class BasicDeConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.PReLU, upsample=False):
        super(GenBlock, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.b1 = nn.BatchNorm2d(in_channels, affine=False)
        self.act1 = activation()
        self.b2 = nn.BatchNorm2d(hidden_channels, affine=False)
        self.act2 = activation()
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self._initialize()

    def _initialize(self):
        import math
        nn.init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            nn.init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.act1(h)
        h = _upsample(self.c1(h)) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = _upsample(self.c_sc(x)) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)






class One_Layer_Attacker(nn.Module):
    def __init__(self, eps=(8 / 255.) / std, type='inf', imsize=32, input_channel=6, bounds = (0.,1.)):
        super(One_Layer_Attacker, self).__init__()
        self.res_group = nn.Sequential(
            BasicConv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            GenBlock(64,64),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)
        )

        if type=='inf':
            self.tanh = nn.Tanh()
        self.eps = eps
        self.type = type
        self.imsize = imsize
        self.input_channel = input_channel

        # image normalization
        self._min = bounds[0]
        self._max = bounds[1]

        self.scale_last = Parameter(torch.Tensor(1))
        self.scale_last.data *= 0
        self.scale_last.data += 5

    def forward(self, x):
        p_x = self.res_group(x)

        if self.type==2:
            # normalize input_norm
            p_x_norm = ((p_x**2).view(-1,self.input_channel*self.imsize**2).mean(dim=1)**0.5).clamp(min = 1e-8).view(-1,1,1,1)
            p_x = p_x/p_x_norm
            p_x = p_x * self.eps
        elif self.type=='inf':
            p_x = p_x.tanh()
            p_x = p_x * self.eps
        else:
            raise RuntimeError("self.type = "+str(self.type)+"is not implemented")

        return p_x





















