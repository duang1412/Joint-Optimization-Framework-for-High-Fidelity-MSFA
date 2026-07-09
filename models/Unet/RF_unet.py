import torch
import torch.nn as nn
import torch.nn.functional as F
# F里面存着各种激励函数，进去一个Variable，出来一个Variable，最常用的当然就是F.relu(A_VARIABLE)
import  matplotlib as plt
from .network_module2 import *
from .PixelUnShuffle import PixelUnShuffle
# from MAN_arch import *
from .net_res5 import *
from .upsample import Dual_UpSample,UpSample

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
class SGN(nn.Module):
    def __init__(self, opt):
        super(SGN, self).__init__()

        self.main1 = Conv2dLayer(opt.in_channels, opt.start_channels * (2 ** 2), 3, 1, 1, pad_type=opt.pad,
                                 activation=opt.activ, norm=opt.norm)  # 256*256*16----256*256*120
        self.main12 = R2CAB(opt.start_channels * (2 ** 2))  # 256*256*120

        self.bot1 = Conv2dLayer(opt.start_channels * (2 ** 4), opt.start_channels * (2 ** 3), 3, 1, 1, pad_type=opt.pad,
                                activation=opt.activ, norm=opt.norm)  # 128*128*480----128*128*240
        self.bot12 = R2CAB(opt.start_channels * (2 ** 3))  # 128*128*240

        self.mid1 = Conv2dLayer(opt.start_channels * (2 ** 5), opt.start_channels * (2 ** 4), 3, 1, 1, pad_type=opt.pad,
                                activation=opt.activ, norm=opt.norm)  # 64*64*960---64*64*480
        self.mid12 = R2CAB(opt.start_channels * (2 ** 4))  # 64*64*480

        self.mid13 = Conv2dLayer(opt.start_channels * (2 ** 4), opt.start_channels * (2 ** 3), 3, 1, 1,
                                 pad_type=opt.pad, activation=opt.activ, norm=opt.norm)
        self.mid14 = R2CAB(opt.start_channels * (2 ** 3))

        self.bot13 = Conv2dLayer(opt.start_channels * (2 ** 3), opt.start_channels * 4, 3, 1, 1,
                                 pad_type=opt.pad, activation=opt.activ, norm=opt.norm)
        self.bot14 = R2CAB(opt.start_channels * 4)

        self.main4 = Conv2dLayer(opt.start_channels * 4, opt.out_channels, 3, 1, 1, pad_type=opt.pad,
                                 activation=opt.activ, norm=opt.norm)  # 【32——>31】

        # self.conv1 = Conv2dLayer(opt.start_channels * (2 ** 2),opt.start_channels * 4,1,1,0,pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.conv2 = Conv2dLayer(opt.start_channels * (2 ** 3), opt.start_channels * 4, 1, 1, 0, pad_type=opt.pad,
                                 activation=opt.activ, norm=opt.norm)
        self.conv3 = Conv2dLayer(opt.start_channels * (2 ** 4), opt.start_channels * 4, 1, 1, 0, pad_type=opt.pad,
                                 activation=opt.activ, norm=opt.norm)

        self.scm1 = MultiScaleGatedAttn(opt.start_channels * (2 ** 2))
        self.scm2 = MultiScaleGatedAttn(opt.start_channels * (2 ** 3))

        self.mdi = MDI(opt.start_channels * (2 ** 2))
        self.upsample1 = UpSample(opt.start_channels * (2 ** 4),2)
        self.upsample2 = UpSample(opt.start_channels * (2 ** 3),2)

    def forward(self, x):
        x = self.main1(x)  # 256*256*16----256*256*120
        x = self.main12(x)  # 256*256*120
        x = self.main12(x)  # 256*256*120


        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)
        x1 = self.bot1(x1)  # 128*128*480----128*128*240
        x1 = self.bot12(x1)  # 128*128*240
        x1 = self.bot12(x1)  # 128*128*240

        x2 = PixelUnShuffle.pixel_unshuffle(x1, 2)
        x2 = self.mid1(x2)  # 64*64*960---64*64*480

        x2 = self.mid12(x2)  # 64*64*480
        x2 = self.mid12(x2)  # 64*64*480

        x3 = self.upsample1(x2) #128*128*240
        x1_s = self.scm2(x1,x3) # 128*128*240
        x3 = torch.cat((x3,x1_s),1) # 128*128*480
        x3 = self.mid13(x3) # 128*128*480----128*128*240
        x3 = self.mid14(x3)
        x3 = self.mid14(x3)

        x4 = self.upsample2(x3)  # 256*256*120
        x_s = self.scm1(x,x4) # 256*256*120
        x4 = torch.cat((x4,x_s),1) # 256*256*240
        x4 = self.bot13(x4) # 256*256*240----256*256*120
        x4 = self.bot14(x4)
        x4 = self.bot14(x4)

        x2_s = self.conv3(x2)
        x3_s = self.conv2(x3)
        out = self.mdi([x2_s,x3_s,x4],x4)
        out = self.main4(out)  # 从120通道转到16通道

        return out
