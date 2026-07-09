import argparse
import os
import time
from os.path import exists, join

import torch
import torch.nn as nn
from scipy.io import savemat

import models
from utility import *

from torchvision.transforms import Compose  #我自己加的，98行Compose报错

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

prefix = 'DM0'

def _parse_str_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args
def crop_width(data, target_width=1296):
    width,_ , _ = data.shape
    start = (width - target_width) // 2
    return data[ start:start + target_width, :,:]

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(description='Hyperspectral Image Demosaicking.')
    parser.add_argument('--arch', '-a', metavar='ARCH',default='e2e_unet3plus',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2025, help='random seed to use. Default=2025')
    parser.add_argument('--resume', '-r', default=True,action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default='checkpoint/e2e_unet3plus/cave/DM0/cave/model_best_OSP.pth', help='checkpoint to use.')
    parser.add_argument('--test',action='store_true', help='test mode?')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--use-2dconv', default=True,action="store_true", help='whether the network uses 2d convolution?')
    parser.add_argument('--bandwise', default=True,action="store_true", help='whether the network handles the input in a band-wise manner?')
    parser.add_argument('--fac', type=str, default='DM0', help='determine the value of fac in the Softmax layer.')
    parser.add_argument('--dataset', type=str, default='cave', help='determine the testing dataset.')  ###默认是icvl
    parser.add_argument('--no-save', default=True,action='store_true', help='saver mode?')
#action则表示命令行中需要输入，才会发生。
    """Initialization Unet parameters"""
    parser.add_argument('--pad', type=str, default='reflect', help='pad type of networks')
    parser.add_argument('--activ', type=str, default='relu', help='activation type of networks')
    parser.add_argument('--norm', type=str, default='none', help='normalization type of networks')
    parser.add_argument('--in_channels', type=int, default=16, help='input channels for generator')
    parser.add_argument('--out_channels', type=int, default=16, help='output channels for generator')
    parser.add_argument('--start_channels', type=int, default=30, help='start channels for generator')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    print(opt)

    cuda = not opt.no_cuda

    HSI2Tensor = partial(HSI2Tensor, use_2dconv=opt.use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())

    common_transform = lambda x: x

    select_transform = common_transform

    print('==> Preparing data..')

    mat_transforms = Compose([
        select_transform,
        HSI2Tensor()
    ])

    if opt.dataset == 'icvl':
        datadir = '/media/exthdd/datasets/hsi/lzy_data/icvl_101_gt'
        matkey = ''
    elif opt.dataset == 'Harvard':
        datadir = '/media/exthdd/datasets/hsi/lzy_data/harvard_22_gt'
        matkey = ''
    elif opt.dataset == 'cave':
        datadir = 'D:\BaiduNetdiskDownload\dww\CAVE_mat/test'
        matkey = 'imgDouble'
    else:
        print('===== REEOR =====')
    mat_dataset = MatDataFromFolder(datadir, size=None)
    fns = os.listdir(datadir)
    mat_dataset.filenames = [os.path.join(datadir, fn) for fn in fns]

    mat_dataset = TransformDataset(mat_dataset, LoadMatKey(key=matkey))
    if opt.dataset == 'icvl':
        mat_dataset = TransformDataset(mat_dataset, lambda x: crop_width(x, target_width=1008))
    mat_dataset = TransformDataset(mat_dataset, lambda x: x.transpose(2, 0, 1))
    mat_dataset = TransformDataset(mat_dataset, lambda x: ((x - np.min(x))) / (np.max(x) - np.min(x)))

    mat_datasets = ImageTransformDataset(mat_dataset, mat_transforms)
    mat_loaders = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=opt.threads, pin_memory=cuda)

    """Model"""
    print("=> creating model '{}'".format(opt.arch))
    net = models.__dict__[opt.arch](opt)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    if len(opt.gpu_ids) > 1:
        from models.sync_batchnorm import DataParallelWithCallback

        net = DataParallelWithCallback(net, device_ids=opt.gpu_ids)

    if cuda:
        net.cuda()
        criterion.cuda()

    """Resume previous model"""
    # Load checkpoint.
    print('==> Resuming from checkpoint %s..' % opt.resumePath)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.resumePath or './checkpoint/%s/%s/model_best.pth' % (opt.arch, prefix))
    # net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])
    iteration = checkpoint['iteration']
    wavelengths = torch.arange(400, 710, 10).cuda().float()
    def torch2numpy(hsi):
        if opt.use_2dconv:
            R_hsi = hsi.data[0].cpu().numpy()
        else:
            R_hsi = hsi.data[0].cpu().numpy()[0, ...]
        return R_hsi
    """Testing"""

    def test(test_loader):
        net.eval()
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        total_mrae=0

        total_psnr2 = 0
        total_ssim2 = 0
        total_sam2 = 0
        total_mrae2 = 0

        cnt = 0
        sum = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if 'hsup' in opt.arch:
                t = torch.zeros([inputs.shape[0], inputs.shape[1], inputs.shape[2] // 4, inputs.shape[3] // 4])
                for i in range(inputs.shape[1]):
                    t[:, i, :, :] = inputs[:, i, i % 4:inputs.shape[-2]:4, i // 4:inputs.shape[-1]:4]
                inputs = t

            if not opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                if 'e2e' in opt.arch:
                    starttime=time.time()
                    outputs, outputs_demos, Center_wavelength, Bandwidths, outputs_filter,Filter_responses = net(inputs, wavelengths)
                    endtime = time.time()
                else:
                    outputs, weight, mosaic = net(inputs)

            if batch_idx != 0:
                usedtime = endtime-starttime
                sum+=usedtime
            outputs = outputs.cpu()
            outputs_demos = outputs_demos.cpu()
            targets = targets.cpu()
            outputs_filter = outputs_filter.cpu()

            psnr, ssim, sam = MSIQA2(outputs, targets)
            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)
            total_ssim += ssim
            avg_ssim = total_ssim / (batch_idx + 1)
            total_sam += sam
            avg_sam = total_sam / (batch_idx + 1)

            psnr2, ssim2, sam2= MSIQA2(outputs_demos, outputs_filter)
            total_psnr2 += psnr2
            avg_psnr2 = total_psnr2 / (batch_idx + 1)
            total_ssim2 += ssim2
            avg_ssim2 = total_ssim2 / (batch_idx + 1)
            total_sam2 += sam2
            avg_sam2 = total_sam2 / (batch_idx + 1)


            print(batch_idx, len(test_loader),
                  'PSNR: %.4f | SSIM: %.4f | SAM: %.4f | PSNR_demos: %.4f | SSIM_demos: %.4f | SAM_demos: %.4f '
                  % (psnr, ssim, sam, psnr2, ssim2, sam2))

            if opt.no_save:
                filedir = False

            else:
                filedir = 'result/'+opt.arch+'/'+opt.dataset+'/'
            if filedir:
                outpath = join(filedir, fns[cnt])

                cnt += 1

                if not exists(filedir):
                    os.makedirs(filedir)

                if not exists(outpath):
                    savemat(outpath, {'pred': torch2numpy(outputs), 'pnsr': psnr, 'ssim': ssim, 'sam': sam})

        avgtime = sum/10
        print('PSNR: %.4f | SSIM: %.4f | SAM: %.4f | PSNR_demos: %.4f | SSIM_demos: %.4f | SAM_demos: %.4f'  % (avg_psnr, avg_ssim, avg_sam,avg_psnr2, avg_ssim2, avg_sam2))

    test(mat_loaders)


