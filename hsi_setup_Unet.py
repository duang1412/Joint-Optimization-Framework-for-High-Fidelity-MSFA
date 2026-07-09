import os
from os.path import join
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
import torch.nn.functional as F
import models
from utility import *
from utility.indexes import FFTLoss, MultipleLoss, SAMLoss
from utility.metric import PSNR, SAM, SSIM, MetricTracker

from models.Unet.RF_unet import weights_init

loss_type = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    # 'L1-SAM': SAMLoss(),
    'L1-SAM': MultipleLoss([nn.L1Loss(), SAMLoss()], weight=[1, 1e-3]),
    'FFT': FFTLoss()
}
Samloss = SAMLoss()

class Engine(object):
    def __init__(self, prefix, opt):
        self.prefix = prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.para_writer = None
        self.gpu_ids = self.opt.gpu_ids
        self.arch = self.opt.arch
        self.scheduler = None
        # self.softmax_alpha = 1.

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoint', self.opt.arch, self.opt.prefix, self.opt.fac)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_demos_psnr = 0
        self.best_loss = 1e6
        self.best_demos_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        metrics = {'psnr': PSNR}
        metrics2 = {'psnr':PSNR}
        self.metric_tracker = MetricTracker(metrics=metrics)
        self.metric_tracker2 = MetricTracker(metrics=metrics2)

        cuda = not self.opt.no_cuda
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("==> creating model '{}'".format(self.opt.arch))

        self.net = models.__dict__[self.opt.arch](self.opt)


        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)

        self.criterion = loss_type[self.opt.loss]

        if cuda:
            self.net.cuda()

        # self.load_and_extract_weights_from_checkpoint(self.net,'checkpoint/e2e_unet3/cave/DM0/cave/model_best_2028_L1.pth')

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(self.opt.arch, self.opt.prefix)
            self.para_writer = get_summary_writer(self.opt.arch+'_filterpara',self.opt.prefix)

        """Optimization Setup"""
        if 'e2e' in self.arch:
            filter_params = list(map(id, self.net.Specfilter.parameters()))
            base_params = filter(lambda p: id(p) not in filter_params, self.net.parameters())
            self.optimizer = optim.Adam([
                {'params': base_params},
                {'params': self.net.Specfilter.parameters(), 'lr': self.opt.lr2},
            ], lr=self.opt.lr, weight_decay=self.opt.wd)
        else:
            weight_params = list(map(id, self.net.weight.parameters()))
            base_params = filter(lambda p: id(p) not in weight_params, self.net.parameters())
            self.optimizer = optim.Adam([
                {'params': base_params},
                {'params': self.net.weight.parameters(), 'lr': self.opt.lr2},
            ], lr=self.opt.lr, weight_decay=self.opt.wd)

        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, min_lr=self.opt.min_lr,
        #                                    verbose=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.opt.nEpochs, eta_min=self.opt.min_lr)
        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath)
        else:
            print('==> Building model..')

    def __step(self, train, inputs, targets):
        #start_time = time.time()
        if train:
            self.optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss_data = 0
        demosaic_loss_data = 0
        recon_loss_data = 0

        if 'e2e' in self.arch:
            outputs, outputs_demos, Center_wavelength, Bandwidths, outputs_filter, Filter_responses = self.net(inputs,self.wavelengths)
            spectrum_loss = self.net.Specfilter.constraint_loss()
            del(Filter_responses)
            demosaic_loss = self.criterion(outputs_demos, outputs_filter)
        else:
            outputs, weight, mosaic = self.net(inputs)
            spectrum_loss = 0

        recon_loss = self.criterion(outputs, targets)
        loss = recon_loss + 0.7*torch.abs(demosaic_loss-recon_loss) + 0.01*spectrum_loss + 0.001*Samloss(outputs, targets)


        if train:
            loss.backward()
        loss_data += loss.item()
        demosaic_loss_data += demosaic_loss.item()
        recon_loss_data += recon_loss.item()

        if train:
            self.optimizer.step()

        #end_time = time.time()

        C_W = Center_wavelength.tolist()
        B_W = Bandwidths.tolist()
        return outputs, loss_data, demosaic_loss_data,outputs_demos,outputs_filter, C_W, B_W, recon_loss_data

    def load(self, resumePath=None):
        model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_psnr = best_model['psnr']
            self.best_loss = best_model['loss']
            self.best_demos_psnr = best_model['psnr2']
            self.best_demos_psnr = best_model['loss2']

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.last_epoch = self.epoch

    """Training"""

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)

        self.net.train()

        self.metric_tracker.refresh()
        self.metric_tracker2.refresh()
        train_loss = 0
        train_demos_loss = 0
        train_recon31_loss = 0
        self.wavelengths = torch.arange(400, 701, 10).cuda().float()
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs, loss_data, demos_loss, outputs_demosaic, outputs_filterlayer, c_w, b_w, recon31_loss= self.__step(True, inputs, targets)

            self.metric_tracker.update(outputs, targets)
            self.metric_tracker2.update(outputs_filterlayer,outputs_demosaic)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)
            train_demos_loss += demos_loss
            avg_demos_loss = train_demos_loss / (batch_idx + 1)
            train_recon31_loss += recon31_loss
            avg_recon31_loss = train_recon31_loss / (batch_idx + 1)

            if not self.opt.no_log:
                self.writer.add_scalar(join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(join(self.prefix, 'train_demos_loss'), demos_loss, self.iteration)
                self.writer.add_scalar(join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)
                self.writer.add_scalar(join(self.prefix, 'train_avg_demos_loss'), avg_demos_loss, self.iteration)
                self.writer.add_scalar(join(self.prefix, 'train_avg_recon_loss'), avg_recon31_loss, self.iteration)
                if self.iteration % 200 == 0:
                    center_wavelengths_dict = {f'filter_centerwavelength{i}': value for i, value in enumerate(c_w)}
                    bandwidths_dict = {f'filter_bandwidth{i}': value for i, value in enumerate(b_w)}
                    self.para_writer.add_scalars(join(self.prefix, 'centerlengths'), center_wavelengths_dict, self.iteration)
                    self.para_writer.add_scalars(join(self.prefix, 'bandwidths'), bandwidths_dict, self.iteration)
                    self.writer.add_histogram(join(self.prefix, 'centerlengths_histogram'), c_w, self.iteration)
                    self.writer.add_histogram(join(self.prefix, 'bandwidths_histogram'), b_w, self.iteration)

            self.iteration += 1
            if (batch_idx + 1) % 100 == 0:
                print(batch_idx, '/', len(train_loader), 'AvgLoss: %.4e | PSNR: %.4f | demosaic_loss: %.4e | PSNR: %.4f' % (
                avg_loss, self.metric_tracker.get_all()['psnr'],avg_demos_loss, self.metric_tracker2.get_all()['psnr']))
                print('filter_parameters_centerlengths:', self.net.Specfilter.center_wavelengths.data,'\nfilter_parameters_centerlengths:', self.net.Specfilter.bandwidths.data)
        self.epoch += 1

        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'train_recon_loss_epoch'), avg_recon31_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'train_demos_loss_epoch'), avg_demos_loss, self.epoch)

    """Validation"""

    def validate(self, valid_loader):
        self.net.eval()

        self.metric_tracker.refresh()
        validate_loss = 0
        validate_demos_loss=0
        validate_recon_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, demos_loss, outputs_demosaic, outputs_filterlayer, c_w, b_w, recon_loss = self.__step(False, inputs, targets)

                self.metric_tracker.update(outputs, targets)
                self.metric_tracker2.update(outputs_demosaic,outputs_filterlayer)
                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx + 1)
                validate_demos_loss += demos_loss
                avg_demos_loss = validate_demos_loss / (batch_idx + 1)
                validate_recon_loss += recon_loss
                avg_recon_loss = validate_recon_loss / (batch_idx + 1)
                if (batch_idx + 1) % 20 == 0:
                    print(batch_idx, '/', len(valid_loader), 'Loss: %.4e | PSNR: %.4f | demosaic_loss: %.4e | PSNR: %.4f'
                        % (avg_loss, self.metric_tracker.get_all()['psnr'], avg_demos_loss, self.metric_tracker2.get_all()['psnr']))

        avg_psnr = self.metric_tracker.get('psnr')
        avg_demos_psnr = self.metric_tracker2.get('psnr')
        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'val_recon_loss_epoch'), avg_recon_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'val_demos_loss_epoch'), avg_demos_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'val_psnr_epoch'), avg_psnr, self.epoch)
            self.writer.add_scalar(join(self.prefix, 'val_demos_psnr_epoch'), avg_demos_psnr, self.epoch)


        print('filter_parameters_centerlengths:', self.net.Specfilter.center_wavelengths.data,'\nfilter_parameters_centerlengths:', self.net.Specfilter.bandwidths.data)

        """Save checkpoint"""
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, self.prefix, 'model_best_%d.pth' % (self.opt.seed))
            self.save_checkpoint(psnr=avg_psnr, loss=avg_loss, psnr2=avg_demos_psnr,loss2=avg_demos_loss,model_out_path=model_best_path)
            self.best_psnr = avg_psnr
            self.best_demos_psnr = avg_demos_psnr
            self.best_loss = avg_loss
            self.best_demos_loss = avg_demos_loss

        return avg_psnr, avg_loss , avg_demos_psnr, avg_demos_loss

    def save_checkpoint(self, psnr, loss, psnr2, loss2, model_out_path=None):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d_%d.pth" % (self.epoch, self.iteration, self.opt.seed))
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'psnr': psnr,
            'loss': loss,
            'psnr2':psnr2,
            'loss2':loss2,
            'epoch': self.epoch,
            'iteration': self.iteration,
        }

        if not os.path.isdir('checkpoint'):
            os.makedirs('checkpoint')
        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
