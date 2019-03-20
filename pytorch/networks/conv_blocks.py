# Blocks for encoder/decoder parts
import torch.nn as nn
import torch.nn.functional as F
import torch

# Activation function
act = nn.LeakyReLU(0.2, inplace=True)


def get_conv_type(upsample):
    if upsample:
        return nn.ConvTranspose2d
    else:
        return nn.Conv2d


def conv4x4(inplanes, outplanes, upsample=False, padding=1):
    # convolution will be Conv2d, or ConvTranspose2d
    _conv = get_conv_type(upsample)
    return _conv(inplanes, outplanes, kernel_size=4, stride=2, padding=padding)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlockToCrop(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlockToCrop, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True, padding=0)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlockCU(nn.Module):
    # upsample at the end
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlockCU, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv3x3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv4x4(outplanes, outplanes, upsample=True)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlock2(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock2, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropout:
            out = self.dropout3(out)
        out = self.relu3(out)

        return out


class BasicBlock5x5(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock5x5, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = self.conv(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def conv(self, in_planes, out_planes, stride=1):
        "5x5 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                         padding=2, bias=False)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BilinearBlock(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BilinearBlock, self).__init__()
        self.dropout = use_dropout

        # Upsample/ Downsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x3 = conv3x3(inplanes, outplanes)

        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.upsample(x)

        out = self.conv3x3(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class UpsampleBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpsampleBlock, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv4x4(num_input_features, num_output_features, upsample=True))


# class DecoderDenseBlock(nn.Module):
#     def __init__(self, num_init_features, block_config, bn_size=4,
#                  drop_rate=0, growth_rate=32):
#         super(DecoderDenseBlock, self).__init__()

#         num_features = num_init_features
#         for i, num_layers in enumerate(reversed(block_config)):
#             block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                 bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 self.features.add_module('transition%dpool' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
#                 num_features = num_features // 2

#     def forward(self, x):

#         return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('upsample', conv4x4(num_output_features, num_output_features, upsample=True))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def get_decoder_block(block_type):
    if block_type == 'bilinear':
        return BilinearBlock
    if block_type == 'basic':
        return BasicBlock
    if block_type == 'basictocrop':
        return BasicBlockToCrop
    if block_type == 'basic_conv_up':
        return BasicBlockCU
    if block_type == 'basic2':
        return BasicBlock2
    if block_type == 'basic5x5':
        return BasicBlock5x5
    # if block_type == 'dense':
    #     return DenseBlock
    if block_type == 'residual':
        pass
