import argparse
import os

import models
from hsi_setup_Unet import Engine
from utility import *

from torchvision.transforms import Compose  #我自己加的，98行Compose报错

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def _parse_str_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(description='Hyperspectral Image Demosaicking.')
    parser.add_argument('--arch', '-a', metavar='ARCH',default='e2e_unet3plus',choices=model_names,help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int, default=8, help='training batch size. Default=16')
    parser.add_argument('--nEpochs', '-n', type=int, default=100, help='number of epochs to train for. Default=100')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning Rate. Default=1e-4.')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning Rate. Default=1e-1.')
    parser.add_argument('--min-lr', '-mlr', type=float, default=5e-6, help='Minimal Learning Rate. Default=1e-5.')
    parser.add_argument('--ri', type=int, default=100, help='Record interval. Default=1')
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', default=False,action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')      #这里原本是8，修改为1
    parser.add_argument('--seed', type=int, default=2025, help='random seed to use. Default=2025')
    parser.add_argument('--resume', '-r', default=False,action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default='checkpoint/e2e_unet3plus/cave/DM0/cave/model_best_3051.pth', help='checkpoint to use.')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--prefix', '-p', type=str, default='cave', help='distinguish checkpoint')
    parser.add_argument('--datadir', '-d', type=str, default='E:\dww\CAVE64_31_20.db', help='path to training set')
    parser.add_argument('--fac', type=str, default='DM0', help='determine the value of fac in the Softmax layer.')
    parser.add_argument('--loss', type=str, default='L1', help='determine the type of Loss Function.')

    """Initialization Unet parameters"""
    parser.add_argument('--pad', type=str, default='reflect', help='pad type of networks')
    parser.add_argument('--activ', type=str, default='relu', help='activation type of networks')
    parser.add_argument('--norm', type=str, default='none', help='normalization type of networks')
    parser.add_argument('--in_channels', type=int, default=16, help='input channels for generator')
    parser.add_argument('--out_channels', type=int, default=16, help='output channels for generator')
    parser.add_argument('--start_channels', type=int, default=30, help='start channels for generator')  #
    parser.add_argument('--n_feat', type=int, default=16, help='input channels for generator')
    parser.add_argument('--stage', type=int, default=3, help='level')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    print(opt)

    cuda = not opt.no_cuda

    """Setup Engine"""
    engine = Engine(opt.prefix, opt)

    use_2dconv = engine.net.module.use_2dconv if len(opt.gpu_ids) > 1 else engine.net.use_2dconv
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())

    common_transform = lambda x: x
    select_transform = common_transform

    train_transform = Compose([
        select_transform,
        HSI2Tensor()
    ])
    valid_transform = Compose([
        select_transform,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31 = LMDBDataset(opt.datadir)
    """Split patches dataset into training, validation parts"""
    icvl_64_31 = TransformDataset(icvl_64_31, common_transform)

    icvl_64_31_T, icvl_64_31_V = get_train_valid_dataset(icvl_64_31, 8)  # 500 for icvl and harvard, and 8 for cave

    train_dataset = ImageTransformDataset(icvl_64_31_T, train_transform)
    valid_dataset = ImageTransformDataset(icvl_64_31_V, valid_transform)

    icvl_64_31_TL = DataLoader(train_dataset,
                    batch_size=opt.batchSize, shuffle=True,
                    num_workers=opt.threads, pin_memory=cuda)  #创建训练Dataloader实例

    icvl_64_31_VL = DataLoader(valid_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=opt.threads, pin_memory=cuda)  #创建验证Dataloader实例

    
    # adjust_learning_rate(engine.optimizer, opt.lr, opt.lr2)
    while engine.epoch < opt.nEpochs:

        engine.train(icvl_64_31_TL)
        psnr, loss, psnr2, loss2  = engine.validate(icvl_64_31_VL)
            
        # engine.scheduler.step(loss)
        engine.scheduler.step()
        lrs = display_learning_rate(engine.optimizer)
        if engine.epoch % opt.ri == 0:
            engine.save_checkpoint(psnr, loss,psnr,loss2)

