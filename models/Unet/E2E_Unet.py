import time

import torch
import torch.nn as nn

from models.admmn import common
from models.admmn.common import *
from models.Unet import RF_unet,net_res5,network_module2,PixelUnShuffle
from models.Unet.RF_unet import SGN


class EDSR_31(nn.Module):
    def __init__(self, n_resblocks, n_colors, n_feats, conv=common.default_conv):
        super(EDSR_31, self).__init__()

        kernel_size = 3
        res_scale = 0.1
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [conv(n_feats, n_feats, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
class e2e_RF_unet3plus(nn.Module):
    def __init__(self, opt):
        super(e2e_RF_unet3plus, self).__init__()

        kernel_size = 3
        out_channels = 16
        self.Specfilter = SpectralFilterLayer(out_channels)
        # self.MSFA_3DLayer = BTES3D()
        #self.MSFA_3DLayer = SEQ3D()
        self.MSFA_3DLayer = OSPMosaic3D(out_channels)
        self.net1 = SGN(opt)
        self.WB = WB_ConvLayer(out_channels)
        self.net2 = common.ConvADMM(n_convs=1, in_channels=32, out_channels=16, kernel_size=kernel_size)
        self.net3 = EDSR_31(n_resblocks=8, n_colors=out_channels, n_feats=31)
        # self.net3 = common.ConvADMM(n_convs=3, in_channels=16, out_channels=31, kernel_size=kernel_size)

    def forward(self, y, wavelengths):
        y_filter, center_wavelengths, bandwidths, filter_responses = self.Specfilter(y, wavelengths)
        y_mosaic3D,mask = self.MSFA_3DLayer(y_filter)
        # y_mosaic2D = torch.sum(y_mosaic3D,dim=1)
        y_coarse = self.WB(y_mosaic3D)
        y_demosaic = self.net1(y_coarse)
        y_s = torch.cat((y_coarse,y_demosaic),1)
        y_fusion = self.net2(y_s)
        y_recon = self.net3(y_fusion)
        return y_recon, y_fusion, center_wavelengths, bandwidths, y_filter, filter_responses


