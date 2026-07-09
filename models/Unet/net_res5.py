# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from models.Unet.network_module2 import Conv2dLayer
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self,channel,ratio= 16):
        super(SELayer,self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.gap = nn.AdaptiveAvgPool2d(1)          # sq 压缩，自适应平均池化
        self.fc = nn.Sequential(                    # ex 激励，2个fc
                nn.Linear(channel, channel //ratio, bias=False),
                nn.ReLU(inplace = True),
                nn.Linear(channel //ratio, channel, bias=False),
                nn.Sigmoid()
        )
        # nn.Linear(in_features，out_features，bias=False)
# n_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
# out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]。

    def forward(self, x):
        b,c,h,w = x.size()
        y = self.gap(x).view(b,c)           # sq 压缩
        y = self.fc(y).view(b,c,1,1)        # ex 激励
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()

        self.ca = SELayer(channel)
        self.sa = SpatialAttention()
        self.main = nn.Sequential(
            Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu'),
            Conv2dLayer(channel, channel, kernel_size=1, stride=1, padding=0, activation='relu')
        )

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        x_att = x + y
        return self.main(x_att) + x_att

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.sa = Attention(channel)

    def forward(self, xs, anchor):
        b,c,h,w = anchor.size()
        ans = torch.zeros_like(anchor)

        for i, x in enumerate(xs):
            if x.shape[-1] > w:
                x = F.adaptive_avg_pool2d(x, (h,w))
                x = self.sa(x)
            elif x.shape[-1] < w:
                x = F.interpolate(x, size=(h,w),mode='bilinear', align_corners=True)
                x = self.sa(x)
            elif x.shape[-1] == w:
                x = self.sa(x)

            ans = ans + x

        return ans

class MDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.sa = Attention(channel)

    def forward(self, xs, anchor):
        b,c,h,w = anchor.size()
        ans = torch.zeros_like(anchor)

        for i, x in enumerate(xs):
            if x.shape[-1] > w:
                x = F.adaptive_avg_pool2d(x, (h,w))
                # x = self.sa(x)
            elif x.shape[-1] < w:
                x = F.interpolate(x, size=(h,w),mode='bilinear', align_corners=True)
                # x = self.sa(x)
            elif x.shape[-1] == w:
                x = x
                # x = self.sa(x)

            ans = ans + x

        return ans

class SCM(nn.Module):
    def __init__(self,channel):
        super(SCM,self).__init__()
        self.main = nn.Sequential(
            Conv2dLayer(channel,channel//4,kernel_size=3,stride=1,padding=1,activation='relu'),
            Conv2dLayer(channel//4, channel // 2, kernel_size=1, stride=1, padding=0, activation='relu'),
            Conv2dLayer(channel//2, channel // 2, kernel_size=3, stride=1, padding=1, activation='relu'),
            Conv2dLayer(channel//2, channel, kernel_size=1, stride=1, padding=0, activation='relu')
        )

        self.conv = Conv2dLayer(channel * 2,channel,kernel_size=1,stride=1,padding=0,activation='relu')

    def forward(self,x):
        x = torch.cat([x,self.main(x)],1)
        return self.conv(x)


class R2CAB(nn.Module):
    def __init__(self, channel, stride=1, scale=5, basewidth=256):
        super(R2CAB, self).__init__()
        width = int(math.floor(basewidth / scale))
        self.conv1 = nn.Conv2d(channel, width * scale, kernel_size=1, stride=stride, bias=True)     # conv 1*1
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 2    #3
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)
        self.conv3 = nn.Conv2d(width * scale, channel, kernel_size=1, stride=stride, bias=True)      # conv 1*1

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.width = width

        self.conv_fft = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, kernel_size=1, stride=stride, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=1, stride=stride, bias=True)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        # [N, width * scale, H, W]
        out = self.relu(self.conv1(x))   # conv 1*1

        # scale * [N, width , H, W]  ,分组
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)      # 三个conv3*3的合并
        out = torch.cat((out, spx[self.nums]), 1)      # 三个conv3*3和直接下来的合并
        sf = torch.fft.rfft2(spx[-1])  # 对最后一个分组的特征图进行二维快速傅里叶变换，得到频域表示
        sf_im = sf.imag  # 获取频域表示的虚部
        sf_re = sf.real  # 获取频域表示的实部
        sf_f = torch.cat([sf_re, sf_im], 1)  # 将实部和虚部在通道维度上拼接，形成一个新的特征图
        sf = self.conv_fft(sf_f)  # 对拼接后的特征图进行卷积操作
        sf_re, sf_im = torch.chunk(sf, 2, 1)  # 将卷积后的特征图在通道维度上分成两部分，分别对应实部和虚部
        sf = torch.complex(sf_re, sf_im)  # 将分开的实部和虚部重新组合成复数形式
        sf = torch.fft.irfft2(sf, s=(H, W))  # 对复数形式的特征图进行二维逆快速傅里叶变换，回到空间域
        out = torch.cat((out, sf), 1)  # 将原始特征图和经过傅里叶变换处理的特征图在通道维度上拼接
        out = self.conv3(out)  # 对拼接后的特征图进行1x1卷积，调整通道数

        out += residual
        return out

# spatial-spectral domain attention learning(SDL)
class SPA_attention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SPA_attention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.sigmoid = nn.Sigmoid()
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.con = nn.Conv2d(self.inplanes,self.inplanes,kernel_size=5,padding=2,stride=1,bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):


        spa_x = self.conv_v_left(x)
        spa_x = torch.mean(spa_x,dim=1)
        spa_x = torch.unsqueeze(spa_x,dim=1)
        mask_sp = self.sigmoid(spa_x)
        x = self.relu(self.con(x))

        out = x * mask_sp

        return out

class Attention2(nn.Module):
    def __init__(self, channel):
        super(Attention2, self).__init__()

        self.SimAM = Simam_module()
        self.main = nn.Sequential(
            Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu'),
            Conv2dLayer(channel, channel, kernel_size=1, stride=1, padding=0, activation='relu')
        )

    def forward(self, x):
        y = self.SimAM(x) * x
        x_att = x + y
        return self.main(x_att) + x_att

class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return  x * self.act(y)


##嵌套注意力融合模块
class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Conv2d(2, 1, 1, 1)

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)
        x2 = self.maxpool(x_)
        cat = torch.cat((x, x2), dim=1)
        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        self.reduction = 1 if reduction is None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = nn.Conv2d(dim, dim // self.reduction, kernel_size=1)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        return dconv

    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()

    def forward(self, x, g):
        x = self.local(x)
        g = self.global_(g)
        fuse = x + g
        return fuse


class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.conv_block = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()
        multi = self.multi(x, g)
        multi = self.selection(multi)
        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)
        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_
        x_att = x_att + x_
        g_att = g_att + g_
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att
        interaction = x_att_2 * g_att_2
        projected = torch.sigmoid(self.proj(interaction))
        weighted = projected * x_
        y = self.conv_block(weighted)
        return y