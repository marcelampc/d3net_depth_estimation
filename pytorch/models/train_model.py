# Based on cycleGAN

import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import _pickle as pickle
from math import sqrt

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import networks.networks as networks
from util.visualizer import Visualizer

# New losses super cool amazing
from util.loss_bank import BerHuLoss, EigenLoss, EigenGradLoss
from util.loss_bank import HuberLoss, RankLoss, L1LogLoss
from util.loss_bank import CauchyLoss, MSEScaledError
from util.pytorch_ssim import SSIM

# import gc
from ipdb import set_trace as st


class TrainModel():
    def name(self):
        return 'Train Model'

    def initialize(self, opt):
        self.opt = opt
        self.opt.imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        self.gpu_ids = ''
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints,
                                             self.opt.name)
        self.scheduler = None
        self.create_save_folders()

        # criterion to evaluate the val split
        self.criterion_eval = MSEScaledError()
        self.mse_scaled_error = MSEScaledError()

        self.opt.print_freq = self.opt.display_freq

        self.visualizer = Visualizer(opt)

        if self.opt.resume and self.opt.display_id > 0:
            self.load_plot_data()
        elif opt.train:
            self.start_epoch = 1
            self.best_val_error = 999.9
        # self.print_save_options()

        # Logfile
        self.logfile = open(os.path.join(self.checkpoints_path,
                                         'logfile.txt'), 'a')
        if opt.validate:
            self.logfile_val = open(os.path.join(self.checkpoints_path,
                                                 'logfile_val.txt'), 'a')

        # Prepare a random seed that will be the same for everyone
        # opt.manualSeed = random.randint(1, 10000)   # fix seed
        # print("Random Seed: ", opt.manualSeed)
        # # random.seed(opt.manualSeed)
        # torch.manual_seed(opt.manualSeed)

        self.random_seed = 123
        random.seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.manual_seed(self.random_seed)
        if opt.cuda:
            self.cuda = torch.device('cuda:0') # set externally. ToDo: set internally
            torch.cuda.manual_seed(self.random_seed)

        # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
        cudnn.benchmark = self.opt.use_cudnn_benchmark  # using too much memory - use when not in astroboy
        cudnn.enabled = True

        if not opt.train and not opt.test and not opt.resume:
            raise Exception("You have to set --train or --test")

        if torch.cuda.is_available and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should run WITHOUT --cpu")
        if not torch.cuda.is_available and opt.cuda:
            raise Exception("No GPU found, run WITH --cpu")

    def set_input(self, input):
        self.input = input

    def create_network(self):
        netG = networks.define_G(input_nc=self.opt.input_nc,
                                 output_nc=self.opt.output_nc, ngf=64,
                                 net_architecture=self.opt.net_architecture,
                                 opt=self.opt,
                                 gpu_ids='')

        if self.opt.cuda:
            netG = netG.cuda()
        return netG

    def get_optimizerG(self, network, lr, weight_decay=0.0):
        generator_params = filter(lambda p: p.requires_grad, network.parameters())
        return optim.Adam(generator_params, lr=lr, betas=(self.opt.beta1, 0.999), weight_decay=weight_decay)

    def get_checkpoint(self, epoch):
        pass

    def train_batch(self):
        """Each method has a different implementation"""
        pass

    def display_gradients_norms(self):
        return 'nothing yet'

    def get_current_errors_display(self):
        pass

    def get_regression_criterion(self):
        if self.opt.regression_loss == 'L1':
            return nn.L1Loss()

    def get_variable(self, tensor, requires_grad=False):
        if self.opt.cuda:
            tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad)

    def restart_variables(self):
        self.it = 0
        self.rmse = 0
        self.n_images = 0

    def train(self, data_loader, val_loader=None):
        self.data_loader = data_loader
        self.len_data_loader = len(self.data_loader)    # check if gonna use elsewhere
        self.total_iter = 0
        for epoch in range(self.start_epoch, self.opt.nEpochs):
            self.restart_variables()
            self.data_iter = iter(self.data_loader)
            # self.pbar = tqdm(range(self.len_data_loader))
            self.pbar = range(self.len_data_loader)
            # while self.it < self.len_data_loader:
            for self.it in self.pbar:
                if self.opt.optim == 'SGD':
                    self.scheduler.step()

                self.total_iter += self.opt.batchSize
                
                self.netG.train(True)

                iter_start_time = time.time()

                self.train_batch()

                d_time = (time.time() - iter_start_time) / self.opt.batchSize

                # print errors
                self.print_current_errors(epoch, d_time)

                # display errors
                self.display_current_results(epoch)

                # Validate
                self.evaluate(val_loader, epoch)

            # save checkpoint
            self.save_checkpoint(epoch, is_best=0)

        self.logfile.close()

        if self.opt.validate:
            self.logfile_val.close()

    def get_next_batch(self):
        # self.it += 1 # important for GANs
        rgb_cpu, depth_cpu = self.data_iter.next()
        # depth_cpu = depth_cpu[0]
        self.input.data.resize_(rgb_cpu.size()).copy_(rgb_cpu)
        # self.target.data.resize_(depth_cpu.size()).copy_(depth_cpu)

    def apply_valid_pixels_mask(self, *data, value=0.0):
        # self.nomask_outG = data[0].data   # for displaying purposes
        mask = (data[1].data > value).to(self.cuda, dtype=torch.float32)
        
        masked_data = []
        for d in data:
            masked_data.append(d * mask)

        return masked_data, mask.sum()

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter_decay and self.opt.use_cgan:  # but independs if conditional or not
            # Linear decay for discriminator
            [self.opt.d_lr, self.optimD] = self._update_learning_rate(self.opt.niter_decay, self.opt.d_lr, self.optimD)
            [self.opt.lr, self.optimG] = self._update_learning_rate(self.opt.niter_decay, self.opt.lr, self.optimG)

    def _update_learning_rate(self, niter_decay, old_lr, optim):
        lr = old_lr - old_lr / niter_decay
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        return lr, optim

    # CONTROL FUNCTIONS OF THE ARCHITECTURE

    def _get_plot_data_filename(self, phase):
        return os.path.join(self.checkpoints_path,
                            'plot_data' + ('' if phase == 'train' else '_' + phase) + '.p')

    def save_static_plot_image():
        return None

    def save_interactive_plot_image():
        return None

    def _save_plot_data(self, plot_data, filename):
        # save
        pickle.dump(plot_data, open(filename, 'wb'))

    def save_plot_data(self):
        self._save_plot_data(self.visualizer.plot_data,
                             self._get_plot_data_filename('train'))
        if self.opt.validate and self.total_iter > self.opt.val_freq:
            self._save_plot_data(self.visualizer.plot_data_val,
                                 self._get_plot_data_filename('val'))

    def _load_plot_data(self, filename):
        # verify if file exists
        if not os.path.isfile(filename):
            raise Exception('In _load_plot_data file {} doesnt exist.'.format(filename))
        else:
            return pickle.load(open(filename, "rb"))

    def load_plot_data(self):
        self.visualizer.plot_data = self._load_plot_data(self._get_plot_data_filename('train'))
        if self.opt.validate:
            self.visualizer.plot_data_val = self._load_plot_data(self._get_plot_data_filename('val'))

    def save_checkpoint(self, epoch, is_best):
        if epoch % self.opt.save_checkpoint_freq == 0 or is_best:
            checkpoint = self.get_checkpoint(epoch)
            checkpoint_filename = '{}/{:04}.pth.tar'.format(self.checkpoints_path, epoch)
            self._save_checkpoint(checkpoint, is_best=is_best, filename=checkpoint_filename)    # standart is_best=0 here cause we didn' evaluate on validation data
            # save plot data as well

    def _save_checkpoint(self, state, is_best, filename):
        print("Saving checkpoint...")
        # uncomment next 2 lines if we still want per epoch
        torch.save(state, filename)
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'latest.pth.tar'))

        # comment next 2 lines if necessary if using last two lines
        # filename = os.path.join(self.checkpoints_path, 'latest.pth.tar')
        # torch.save(state, os.path.join(self.checkpoints_path, 'latest.pth.tar'))

        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoints_path, 'best.pth.tar'))

    def create_save_folders(self):
        if self.opt.train:
            os.system('mkdir -p {0}'.format(self.checkpoints_path))
        # if self.opt.save_samples:
        #     subfolders = ['input', 'target', 'results', 'output']
        #     self.save_samples_path = os.path.join('results/train_results/', self.opt.name)
        #     for subfolder in subfolders:
        #         path = os.path.join(self.save_samples_path, subfolder)
        #         os.system('mkdir -p {0}'.format(path))
        #     if self.opt.test:
        #         self.save_samples_path = os.path.join('results/test_results/', self.opt.name)
        #         self.save_samples_path = os.path.join(self.save_samples_path, self.opt.epoch)
        #         for subfolder in subfolders:
        #             path = os.path.join(self.save_samples_path, subfolder)
        #             os.system('mkdir -p {0}'.format(path))

    def print_save_options(self):
        options_file = open(os.path.join(self.checkpoints_path,
                                         'options.txt'), 'w')
        args = dict((arg, getattr(self.opt, arg)) for arg in dir(self.opt) if not arg.startswith('_'))
        print('---Options---')
        for k, v in sorted(args.items()):
            option = '{}: {}'.format(k, v)
            # print options
            print(option)
            # save options in file
            options_file.write(option + '\n')

        options_file.close()

    def mean_errors(self):
        pass

    def get_current_errors(self):
        pass

    def print_current_errors(self, epoch, d_time):
        if self.total_iter % self.opt.print_freq == 0:
            self.mean_errors()
            errors = self.get_current_errors()
            message = self.visualizer.print_errors(errors, epoch, self.it,
                                            self.len_data_loader, d_time)

            # self.pbar.set_description(message)
            print(message)
        # self.pbar.refresh()

    # def print_epoch_error(error):
    #     pass

    def get_current_visuals(self):
        pass

    def display_current_results(self, epoch):
        if self.opt.display_id > 0 and self.total_iter % self.opt.display_freq == 0:

            errors = self.get_current_errors_display()
            self.visualizer.display_errors(errors, epoch,
                                           float(self.it) / self.len_data_loader)

            visuals = self.get_current_visuals()

            self.visualizer.display_images(visuals, epoch)

            # save printed errors to logfile
            self.visualizer.save_errors_file(self.logfile)

    def evaluate(self, data_loader, epoch):
        if self.opt.validate and self.total_iter % self.opt.val_freq == 0:
            val_error = self.get_eval_error(data_loader, self.netG,
                                            self.criterion_eval, epoch)

            # errors = OrderedDict([('LossL1', self.e_reg if self.opt.reg_type == 'L1' else self.L1error),
            #                      ('ValError', val_error.item())])
            errors = OrderedDict([('RMSE', self.rmse_epoch), ('RMSEVal', val_error)])
            self.visualizer.display_errors(errors, epoch, float(self.it) / self.len_data_loader, phase='val')
            message = self.visualizer.print_errors(errors, epoch, self.it, len(data_loader), 0)
            print('[Validation] ' + message)
            self.visualizer.save_errors_file(self.logfile_val)
            self.save_plot_data()
            # save best models
            is_best = self.best_val_error > val_error
            if is_best:     # and not self.opt.not_save_val_model:
                print("Updating BEST model (epoch {}, iters {})\n".format(epoch, self.total_iter))
                self.best_val_error = val_error
                self.save_checkpoint(epoch, is_best)

    def get_eval_error(self, val_loader, model, criterion, epoch):
        """
        Validate every self.opt.val_freq epochs
        """
        # no need to switch to model.eval because we want to keep dropout layers. Do I gave to ignore batch norm layers?
        cumulated_rmse = 0
        batchSize = 1
        input = self.get_variable(torch.FloatTensor(batchSize, 3, self.opt.imageSize[0], self.opt.imageSize[1]), requires_grad=False)
        mask = self.get_variable(torch.FloatTensor(batchSize, 1, self.opt.imageSize[0], self.opt.imageSize[1]), requires_grad=False)
        target = self.get_variable(torch.FloatTensor(batchSize, 1, self.opt.imageSize[0], self.opt.imageSize[1]))
        # model.eval()
        model.train(False)
        pbar_val = tqdm(val_loader)
        for i, (rgb_cpu, depth_cpu) in enumerate(pbar_val):
            pbar_val.set_description('[Validation]')
            input.data.resize_(rgb_cpu.size()).copy_(rgb_cpu)
            target.data.resize_(depth_cpu.size()).copy_(depth_cpu)

            if self.opt.use_padding:
                from torch.nn import ReflectionPad2d

                self.opt.padding = self.get_padding_image(input)

                input = ReflectionPad2d(self.opt.padding)(input)
                target = ReflectionPad2d(self.opt.padding)(target)

            # get output of the network
            with torch.no_grad():
                outG = model.forward(input)
            # apply mask
            nomask_outG = outG.data   # for displaying purposes
            mask_ByteTensor = self.get_mask(target.data)
            mask.data.resize_(mask_ByteTensor.size()).copy_(mask_ByteTensor)
            outG = outG * mask
            target = target * mask
            cumulated_rmse += sqrt(criterion(outG, target, mask, no_mask=False))

            if(i == 1):
                self.visualizer.display_images(OrderedDict([('input', input.data),
                                                            ('gt', target.data),
                                                            ('output', nomask_outG)]), epoch='val {}'.format(epoch), phase='val')

        return cumulated_rmse / len(val_loader)

    def get_mask(self, data, value=0.0):
        return (target.data > 0.0)

    def get_padding(self, dim):
        final_dim = (dim // 32 + 1) * 32
        return final_dim - dim

    def get_padding_image(self, img):
        # get tensor dimensions
        h, w = img.size()[2:]
        w_pad, h_pad = self.get_padding(w), self.get_padding(h)

        pwr = w_pad // 2
        pwl = w_pad - pwr
        phb = h_pad // 2
        phu = h_pad - phb

        # pwl, pwr, phu, phb
        return (pwl, pwr, phu, phb)

    def adjust_learning_rate(self, initial_lr, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = initial_lr * (0.1 ** (epoch // self.opt.niter_decay))
        if epoch % self.opt.niter_decay == 0:
            print("LEARNING RATE DECAY HERE: lr = {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
