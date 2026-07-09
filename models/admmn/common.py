import math

import numpy as np
import torch
import torch.nn as nn
from utility.util import generate_SP

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ConvADMM(nn.Sequential):
    def __init__(
        self, n_convs, in_channels, out_channels, kernel_size, bias=True,
        bn=False, act=nn.ReLU(True)):

        m = []
        for i in range(n_convs):
            if (i > 0 and in_channels != out_channels):
                in_channels = out_channels
            m.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding=(kernel_size//2), bias=bias)
            )
            if bn:
                m.append(nn.BatchNorm2d(out_channels))
            if act is not None:
                m.append(act)

        super(ConvADMM, self).__init__(*m)
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):#n_feats:特征数features

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    #scale是2的幂。这是因为二进制中，2的幂与其前一个数进行按位与操作时，结果为0
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class MosaicAlpha2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(MosaicAlpha2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.ones([1, self.channel, self.pattern_size, self.pattern_size]), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, alpha):
        w = self.softmax(alpha * self.weight)
        w = w.repeat(1, 1, x.shape[-2]//w.shape[-2], x.shape[-1]//w.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w


class MosaicSpectrum2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(MosaicSpectrum2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.full([1, self.channel, self.pattern_size, self.pattern_size], 1/31), requires_grad=True)  # bs,c,h,w
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        eta3 = 1e-3
        loss1 = eta3 * torch.norm(self.weight, p=2) ** 2  #2范数约束
        eta4 = 1e-2
        loss2 = eta4 * torch.norm(self.weight[:, 1:, :, :] - self.weight[:, :-1, :, :], p=2) ** 2  #平滑约束

        # if torch.min(self.weight) < 0:
        #     w = self.softmax(self.weight)
        # else:
        #     w = self.weight
        # if torch.min(self.weight) < 0:
        #     w = self.weight - torch.min(self.weight)
        # else:
        #     w = self.weight

        zero = torch.zeros_like(self.weight)
        w = torch.where(self.weight < 0, zero, self.weight)  #非负约束
        # w = self.weight
        w = w.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])

        # non_zero_weight = torch.where(self.weight < 0, torch.zeros_like(self.weight), self.weight)
        # w = non_zero_weight.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])

        out = torch.sum(x * w, dim=1)
        
        return out, w, loss1 + loss2


class Mosaic2DBase(nn.Module):
    def __init__(self, pattern_size=4):
        super(Mosaic2DBase, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :]  = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :]  = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :]  = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :]  = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :]  = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :]  = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w


class Mosaic3DBase(nn.Module):
    def __init__(self, pattern_size=4):
        super(Mosaic3DBase, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :] = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :] = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :] = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :] = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :] = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])
        out = torch.sum(x * w, dim=2)
        return out, w


class SSRMosaicAlpha2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(SSRMosaicAlpha2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.ones([1, self.channel, self.pattern_size, self.pattern_size]), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, alpha):
        w = self.softmax(alpha * self.weight)
        w = w.repeat(1, 1, x.shape[-2]//w.shape[-2], x.shape[-1]//w.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w

class SpectralFilterLayer(nn.Module):
    def __init__(self, out_channels, min_wavelength=400, max_wavelength=700, min_bandwidth=10, max_bandwidth=50,
                 sigma=0.01):
        super(SpectralFilterLayer, self).__init__()
        torch.manual_seed(2022) #default=2021
        self.out_channels = out_channels

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.sigma = sigma

        # 定义可训练的中心波长和半峰带宽参数
        # self.center_wavelengths = nn.Parameter(woyuandiama
        #     torch.rand(out_channels) * (max_wavelength - min_wavelength) + min_wavelength, requires_grad=True)
        # self.center_wavelengths = nn.Parameter(torch.arange(407, 700, 19).float(), requires_grad=True)
        self.center_wavelengths = nn.Parameter(torch.arange(400,701,20).float(), requires_grad=True)
        # self.bandwidths = nn.Parameter(
        #     torch.rand(out_channels) * (max_bandwidth - min_bandwidth) + min_bandwidth, requires_grad=True)
        self.bandwidths = nn.Parameter(torch.full((out_channels,),20,dtype=torch.float32), requires_grad=True)

    def forward(self, x, wavelengths):
        batch_size, in_channels, height, width = x.size()
        # 应用滤波器
        center_wavelengths = self.center_wavelengths
        bandwidths = self.bandwidths

        filter_response = torch.zeros(self.out_channels, in_channels, device=x.device)

        for i in range(self.out_channels):
            center_wavelength = center_wavelengths[i]
            bandwidth = bandwidths[i]
            filter_response[i] = torch.exp(-0.5 * ((wavelengths - center_wavelength) / (bandwidth/2.355)) ** 2)
        output = torch.einsum('bchw,oc->bohw', x, filter_response)

        return output, center_wavelengths, bandwidths, filter_response

    def constraint_loss(self):
        # 计算中心波长和半峰带宽的约束损失
        wavelength_loss = torch.mean(
            torch.max(torch.clamp(1 - (self.center_wavelengths - self.min_wavelength) / self.sigma, min=0.0),
                              torch.clamp(1 - (self.max_wavelength - self.center_wavelengths) / self.sigma, min=0.0)))
        bandwidth_loss = torch.mean(
            torch.max(torch.clamp(1 - (self.bandwidths - self.min_bandwidth) / self.sigma, min=0.0),
                         torch.clamp(1 - (self.max_bandwidth - self.bandwidths) / self.sigma, min=0.0)))
        return (wavelength_loss + bandwidth_loss) / 2

class OSPMosaic2D(nn.Module):
    def __init__(self,out_channels):
        super(OSPMosaic2D, self).__init__()
        self.pattern_size = out_channels
        self.MSFA_pattern = np.float32(generate_SP(out_channels, False))
        self.weight = torch.tensor(self.MSFA_pattern).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2] // self.weight.shape[-2], x.shape[-1] // self.weight.shape[-1])
        out_2D = torch.sum(x * w, dim=1)
        return out_2D

class OSPMosaic3D(nn.Module):
    def __init__(self,out_channels):
        super(OSPMosaic3D, self).__init__()
        self.pattern_size = out_channels
        self.MSFA_pattern = np.float32(generate_SP(out_channels))
        self.weight = torch.tensor(self.MSFA_pattern).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2] // self.weight.shape[-2], x.shape[-1] // self.weight.shape[-1])
        out_3d = x * w
        return out_3d, w


class SEQ3D(nn.Module):
    def __init__(self, pattern_size=4):
        super(SEQ3D, self).__init__()
        torch.manual_seed(2021)
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :] = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :] = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :] = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :] = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :] = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2] // self.weight.shape[-2], x.shape[-1] // self.weight.shape[-1])
        out = x * w
        return out

class BTES3D(nn.Module):
    def __init__(self, pattern_size=4):
        super(BTES3D, self).__init__()
        torch.manual_seed(2021)
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[2, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :] = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :] = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        MSFA[6, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[8, :, :] = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2] // self.weight.shape[-2], x.shape[-1] // self.weight.shape[-1])
        out = x * w
        return out

class WB_ConvLayer(nn.Module):
    def __init__(self,channels_number=16,padding_type='reflect'):
        super(WB_ConvLayer, self).__init__()
        # 使用反射填充，卷积核为固定的 Filter_a
        self.conv = nn.Conv2d(in_channels=channels_number, out_channels=channels_number, kernel_size=7, padding=3, padding_mode=padding_type,
                              groups=16,bias=False)
        # 固定卷积核的权重
        self.conv.weight = nn.Parameter(torch.tensor(
            [[1 / 16, 2 / 16, 3 / 16, 4 / 16, 3 / 16, 2 / 16, 1 / 16],
             [2 / 16, 4 / 16, 6 / 16, 8 / 16, 6 / 16, 4 / 16, 2 / 16],
             [3 / 16, 6 / 16, 9 / 16, 12 / 16, 9 / 16, 6 / 16, 3 / 16],
             [4 / 16, 8 / 16, 12 / 16, 16 / 16, 12 / 16, 8 / 16, 4 / 16],
             [3 / 16, 6 / 16, 9 / 16, 12 / 16, 9 / 16, 6 / 16, 3 / 16],
             [2 / 16, 4 / 16, 6 / 16, 8 / 16, 6 / 16, 4 / 16, 2 / 16],
             [1 / 16, 2 / 16, 3 / 16, 4 / 16, 3 / 16, 2 / 16, 1 / 16]]
        ).repeat(16, 1, 1, 1).cuda(), requires_grad=False)

    def forward(self, x):
        return self.conv(x)

class Mosaic2cube(nn.Module):
    def __init__(self, pattern_size=4):
        super(Mosaic2cube, self).__init__()
        torch.manual_seed(2021)

        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :] = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :] = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :] = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :] = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :] = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).cuda()

    def forward(self, x):
        out=torch.zeros((x.shape[0],16,x.shape[-2],x.shape[-1])).cuda()
        w = self.weight.repeat(1, 1, x.shape[-2] // self.weight.shape[-2], x.shape[-1] // self.weight.shape[-1])
        for i in range(16):
            out[:,i,:,:] = x[:,0,:,:]*w[:,i,:,:]
        return out, w