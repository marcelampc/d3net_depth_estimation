# !/usr/bin/env python
# Arguments for cleanCGAN.py

import argparse


class TrainOptions():
    def initialize(self):
        self.parser = argparse.ArgumentParser()
        ########################### General options ###########################
        self.parser.add_argument('--dataroot',
                                 required=True,
                                 action='store', help='path to dataset')
        self.parser.add_argument('--name', required=True, help='name of the test')
        ############################ Data options #############################
        self.parser.add_argument('--imageSize', nargs='+', default=[256], type=int, help='order: width height')  # todo: change
        self.parser.add_argument('--outputSize', nargs='+', default=0, type=int, help='order: width height')  # todo: change
        ########################## Training options ###########################
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--visualize', action='store_true')
        self.parser.add_argument('--test_split', default='test')
        self.parser.add_argument('--train_split', default='train')
        self.parser.add_argument('--lam', type=float, default=100.0)
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_nc', type=int, default=1)
        self.parser.add_argument('--cpu', dest='cuda', action='store_false',
                                 help='Use cpu instead of gpu (default: use gpu)')
        self.parser.set_defaults(cuda=True)
        self.parser.add_argument('--nEpochs', type=int, default=350)
        self.parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                            help='number of data loading threads (default: 2)')
        self.parser.add_argument('--batchSize', '-b', default=16, type=int, metavar='N',
                            help='mini-batch size (1 = pure stochastic) Default: 256')
        self.parser.add_argument('--resume', action='store_true')
        self.parser.add_argument('--epoch', default='latest', help='Resume training or test with best model or last model')
        self.parser.add_argument('--no_mask', action='store_true')
        self.parser.add_argument('--mask_thres', type=float, default=0.0)
        self.parser.add_argument('--update_lr', action='store_true')
        self.parser.add_argument('--init_method', default='normal')
        self.parser.add_argument('--use_cudnn_benchmark', action='store_true')
        ########################## Data augmentation ##########################
        self.parser.add_argument('--data_augmentation', nargs='+', default=["t", "f", "f", "f", "t"], help="hflip vflip scale color rotation")
        ########################### Display options ###########################
        self.parser.add_argument('--display', action='store_true',
                            help='display results (default: false)')
        self.parser.add_argument('--port', type=int, default=8097, help='Display port')
        self.parser.add_argument('--display_id', type=int, default=0)
        self.parser.add_argument('--display_freq', type=int, default=50)
        self.parser.add_argument('--print_freq', type=int, default=50)
        ######################## Optimization options #########################
        self.parser.add_argument('--lr', default=0.0002, type=float, metavar='LR',
                            help='initial learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
        self.parser.add_argument('--weightDecay', default=0.0, type=float, metavar='W',
                            help='weight decay')
        self.parser.add_argument('--optim', default='Adam')
        self.parser.add_argument('--momentum', type=float,  default=0.9)
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        ######################### Checkpoint options ##########################
        self.parser.add_argument('--checkpoints', default='./checkpoints',
                            help='where models are saved (default: ./checkpoints)')
        self.parser.add_argument('--save_samples_freq', type=int, default=10,
                            help='frequency to save samples')
        self.parser.add_argument('--save_samples', action='store_true',
                            help='save examples during training (default: false)')
        self.parser.add_argument('--save_checkpoint_freq', default=30, type=int, help='in epochs')
        ######################### Validation options ##########################
        self.parser.add_argument('--validate', action='store_true')
        self.parser.add_argument('--val_split', default='val')
        self.parser.add_argument('--val_freq', type=int, default=1000, help='number of iterations to validate')
        self.parser.add_argument('--not_save_val_model', action='store_false', help='saves best model from validation')
        ########################### Model options ###########################
        self.parser.add_argument('--pretrained', action='store_true')
        self.parser.add_argument('--pretrained_path', default='no_path', help='path to a pretrained network we want to use')
        self.parser.add_argument('--finetuning', action='store_true')
        self.parser.add_argument('--model', default='regression')
        # self.parser.add_argument('--which_model_netG', default='unet_256')
        self.parser.add_argument('--net_architecture', default='DenseUNet')
        self.parser.add_argument('--d_block_type', default='basic')
        self.parser.add_argument('--use_softmax', action='store_true')
        self.parser.add_argument('--n_classes', type=int, default=1)
        self.parser.add_argument('--use_skips', action='store_true')
        # to implement:
        self.parser.add_argument('--use_dropout', action='store_true')
        self.parser.add_argument('--ngpu', type=int, default=0)
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
        self.parser.add_argument('--annotation', action='store_true', help='alows user to give specific informations about training to readme.txt file')
        self.parser.add_argument('--use_resize', action='store_true')
        # DFF Training
        self.parser.add_argument('--use_crop', action='store_true')
        self.parser.add_argument('--use_padding', action='store_true')
        self.parser.add_argument('--padding', nargs='+', default=[12, 12, 1, 0], type=int, help='widthLeft, widthRight, hUp, heightBottom')
        ############################ Save options ##############################
        self.parser.add_argument('--save_upsample', action='store_true')
        self.parser.add_argument('--upsample_size', nargs='+', type=int, default=[640, 480])
        ########################### Dataset options ############################
        self.parser.add_argument('--dataset_name', default='nyu')
        self.parser.add_argument('--kitti_split', default='eigen_r', help='options are: rob and eigen')
        self.parser.add_argument('--kitti_learning', default='supervised', help='options are: supervised and unsupervised')
        self.parser.add_argument('--scale_to_mm', type=float, default=0.0, help='scale to calculate rmse in meters')
        self.parser.add_argument('--max_distance', type=float, default=10.0)


    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
