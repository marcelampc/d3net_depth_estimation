import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
# import numpy as np
import networks.weight_initialization as w_init

from ipdb import set_trace as st
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, net_architecture, opt, gpu_ids=[]):
    netG = None
    # use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)
    use_dropout = opt.use_dropout
    pretrained = opt.pretrained
    init_method = opt.init_method
    use_skips = opt.use_skips
    d_block_type = opt.d_block_type
    n_classes = opt.n_classes # --> output_nc
    tasks = opt.tasks

    from .dense_decoders_multitask_auto import denseUnet121
    netG = denseUnet121(pretrained=pretrained, input_nc=input_nc, outputs_nc=opt.outputs_nc, init_method=init_method, use_dropout=use_dropout, use_skips=use_skips, d_block_type=d_block_type, num_classes=n_classes, tasks=tasks, type_net=net_architecture)
    # print number of parameters of the network
    # print_network(netG)
    print_n_parameters_network(netG)

    # if len(gpu_ids) > 0:
    #     netG.cuda(device_id=gpu_ids[0])

    if not pretrained:
        w_init.init_weights(netG, init_method)
    return netG


def print_n_parameters_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def print_network(net):
    print(net)
