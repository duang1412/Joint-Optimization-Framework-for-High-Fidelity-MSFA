from .Unet.E2E_Unet import *
"""Define commonly used architecture"""

def e2e_unet3plus(opt):
    net = e2e_RF_unet3plus(opt)
    net.use_2dconv = True
    net.bandwise = False
    return net


