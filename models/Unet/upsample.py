import torch
import torch.nn as nn

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.factor = scale_factor
        if self.factor == 2:
            self.up_p = nn.Sequential(nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//4, in_channels//2, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x_p = self.up_p(x)  # pixel shuffle
        return x_p

