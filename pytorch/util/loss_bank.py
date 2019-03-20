
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

from ipdb import set_trace as st
import numpy as np

from .rank_loss import image_rank_4d

sys.path.append(os.getcwd())


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class RankLoss(Function):
    def __init__(self, rad=5, sigma=3, mean=True, l1=False):
        self.l1 = l1
        self.rad = rad
        self.mean = mean
        self.sigma = sigma
        self.target_rank = None  # mpc: why is it for?

    def forward(self, input, target):
        # ... # implementation
        # print(input.data.type())
        self.type = input.type()

        input_np = input.cpu().numpy()
        target_np = target.cpu().numpy()

        # Get rank image from input
        self.R_input = image_rank_4d(input_np, rank_neighborhood=self.rad,
                                     gaussian_sigma=self.sigma).astype(float) / ((2 * self.rad + 1)**2)
        if self.target_rank is None:
            self.R_target = image_rank_4d(target_np, rank_neighborhood=self.rad,
                                          gaussian_sigma=self.sigma).astype(float) / ((2 * self.rad + 1)**2)
        else:
            self.R_target = self.target_rank.astype(float) / (2 * self.rad + 1)**2

        # self.R_input, mini, maxi = rank_inf_4d_test(input_np,self.rad)
        # self.R_input = (self.R_input*(maxi-mini)).astype(float)/((2*self.rad+1)**2)
        #
        # # self.R_target = image_rank_4d(target_np,rank_neighborhood=self.rad, gaussian_sigma=self.sigma).astype(float)/((2*self.rad+1)**2)
        # if self.target_rank is None:
        #     self.R_target, mini, maxi = rank_inf_4d_test(target_np, self.rad)
        #     self.R_target = (self.R_target*(maxi-mini)).astype(float)/((2*self.rad+1)**2)
        # else:
        #     self.R_target = self.target_rank.astype(float)/(2*self.rad+1)**2

        self.divisor = 1.
        if self.mean:
            self.divisor = self.R_input.size
        if self.l1:
            loss = torch.Tensor([np.abs(self.R_input - self.R_target).sum() / self.divisor]).type(self.type)
        else:
            loss = torch.Tensor([((self.R_input - self.R_target)**2).sum() / self.divisor]).type(self.type)
        return loss   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        # ... # implementation
        grad_input = None

        if self.l1:
            grad_input = ((self.R_input - self.R_target) > 0).astype(float) * 2 - 1
        else:  # l2
            grad_input = 2 * (self.R_input - self.R_target)

        return torch.from_numpy(grad_input / self.divisor).type(self.type), None


class MSEScaledError(nn.Module):
    def __init__(self):
        super(MSEScaledError, self).__init__()

    def forward(self, input, target, no_mask=True):
        # input_meters = ((input + 1) / 2) * self.scale_to_meters
        # target_meters = ((target + 1) / 2) * self.scale_to_meters

        error = input - target

        # print(torch.sum(error * error))
        # print(mask.sum().type(torch.FloatTensor))
        # print(mask.type(torch.FloatTensor).sum())
        # print((mask >= 0.0).sum())
        if no_mask:
            return torch.mean(error * error).cuda()
        else:
            return (torch.sum(error * error)) / mask.sum().type(torch.FloatTensor).cuda()

class BerHuLoss(nn.Module):
    """Adds a Huber Loss term to the training procedure.
    For each value x in `error=labels-predictions`, the following is calculated:
    ```
    |x|                        if |x| <= d
    (x^2 + d^2)*0,5/d  if |x| > d
    ```
    """

    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, input, target):  # delta changes everytime - per batch
        # all variables here must be torch.autograd.Variable to perform autograd
        _assert_no_grad(target)
        absError = torch.abs(target - input)
        delta = 0.2 * torch.max(absError).data[0]

        L2 = (absError * absError / delta + delta) * 0.5

        mask_down_f = absError.le(delta).float()
        mask_up_f = absError.gt(delta).float()

        loss = absError * mask_down_f + L2 * mask_up_f

        return torch.mean(loss)


class HuberLoss(nn.Module):
    """Adds a Huber Loss term to the training procedure.
    For each value x in `error=labels-predictions`, the following is calculated:
    ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    """

    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, input, target):  # delta changes everytime - per batch
        # all variables here must be torch.autograd.Variable to perform autograd
        _assert_no_grad(target)
        error = (target - input)
        absError = torch.abs(error)

        delta = 0.2 * torch.max(absError).data[0]

        ft1 = 0.5 * error * error
        ft2 = 0.5 * delta * delta + delta * (absError - delta)

        mask_down_f = absError.le(delta).float()
        mask_up_f = absError.gt(delta).float()

        loss = ft1 * mask_down_f + ft2 * mask_up_f

        return torch.mean(loss)


class L1LogLoss(nn.Module):
    """
    As we use masks, there are no 0 values.
    """

    def __init__(self):
        super(L1LogLoss, self).__init__()

    def _data_in_log_meters(self, data):
        """ log((0.5 * (data + 1)) * 10) in meters"""
        # data is between [-1, 1]
        # we are going to apply log, but according to the original distance
        return torch.log(5 * (data + 1))

    def forward(self, input, target):  # delta changes everytime - per batch
        # all variables here must be torch.autograd.Variable to perform autograd
        _assert_no_grad(target)

        # get data in log meters
        log_input, log_target = self._data_in_log_meters(input), self._data_in_log_meters(target)

        loss = torch.abs(log_input - log_target)

        return torch.mean(loss)


class CauchyLoss(nn.Module):
    """
    As we use masks, there are no 0 values.
    """

    def __init__(self):
        super(CauchyLoss, self).__init__()

    def forward(self, input, target):  # delta changes everytime - per batch
        # all variables here must be torch.autograd.Variable to perform autograd
        _assert_no_grad(target)

        # get data in log meters
        error = input - target
        l2_loss = torch.mean(error * error)
        loss = torch.log(1 + l2_loss)

        return loss


class EigenLoss(nn.Module):
    def __init__(self):
        super(EigenLoss, self).__init__()

    def _data_in_log_meters(self, data):
        """ log((0.5 * (data + 1)) * 10) in meters"""
        # data is between [-1, 1]
        # we are going to apply log, but according to the original distance
        return torch.log(5 * (data + 1))

    def forward(self, input, target):
        _assert_no_grad(target)
        # get data in log meters
        log_input, log_target = self._data_in_log_meters(input), self._data_in_log_meters(target)

        # number of elements
        n_el = log_input.data.numel()
        error = log_input - log_target

        # L2 loss
        loss1 = torch.mean(error * error)

        # scale invariant difference
        mean_error = torch.mean(error)
        loss2 = (mean_error * mean_error) * 0.5 / (n_el * n_el)

        loss = loss1 - loss2

        return loss


class EigenGradLoss(nn.Module):
    """
    d = log(input) - log(target)
    L()  = (mean(d^2)) - lambda / n^2 * (mean(d)^2)
    """

    def __init__(self, opt):
        super(EigenGradLoss, self).__init__()
        self._lambda = 0.5   # like in Eigen's 2014 paper
        self.mask = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize[0], opt.imageSize[1]), requires_grad=False).cuda()

    def _data_in_log_meters(self, data):
        """ log((0.5 * (data + 1)) * 10) in meters"""
        return torch.log(5 * (data + 1))

    def _sobel_window_x(self):
        return torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def _sobel_window_y(self):
        return -self._sobel_window_x().transpose(0, 1)

    def get_mask_invalid_pixels(self, target, value=-0.9):
        mask_ByteTensor = (target.data > value)
        self.mask.data.resize_(mask_ByteTensor.size()).copy_(mask_ByteTensor)
        # self.outG = self.outG * self.mask
        # self.target = self.target * self.mask

    def forward(self, input, target):
        _assert_no_grad(target)

        self.get_mask_invalid_pixels(target)    # modifies self.mask variable

        # get data in log meters
        log_input = self._data_in_log_meters(input * self.mask)
        log_target = self._data_in_log_meters(target * self.mask)

        # number of elements
        error_image = (log_input - log_target)
        n_el = error_image.data.numel()

        # L2 loss
        loss1 = torch.mean(error_image * error_image)

        # scale invariant difference
        mean_error = torch.mean(error_image)
        loss2 = (mean_error * mean_error) * 0.5 / (n_el * n_el)

        # horizontal and vertical gradients
        # apply sobel filter
        _filter_x = Variable(self._sobel_window_x().unsqueeze(0).unsqueeze(0).cuda())
        _filter_y = Variable(self._sobel_window_y().unsqueeze(0).unsqueeze(0).cuda())

        grad_x = F.conv2d(error_image, _filter_x, padding=1) * self.mask  # use conv2 with predefined weights
        grad_y = F.conv2d(error_image, _filter_y, padding=1) * self.mask
        loss3 = torch.mean(grad_x * grad_x + grad_y * grad_y)

        loss = loss1 - loss2 + loss3

        return loss


class DIWLoss(nn.Module):
    def __init__(self):
        # in criterion: get the relation of distance between the two points
        # at the same positions
        # send relation and get new relation to apply function

        # Also use  mask like in berhu

        super(DIWLoss, self).__init__()

    def _log_loss(self, ptA_pred, ptB_pred):
        return torch.log(1 + torch.exp(ptA_pred - ptB_pred))

    def _mse(self, ptA_pred, ptB_pred):
        return (ptA_pred - ptB_pred) * (ptA_pred - ptB_pred)

    def _get_value(self, output, pt):
        # values_tensor = torch.FloatTensor()
        # for i in range(pt.shape[1]):
        values_tensor = [output[i, 0, pt[1, i] - 1, pt[0, i] - 1] for i in range(pt.shape[1])]
        return values_tensor

    def _create_position_mask(self, output, pt):
        mask = torch.LongTensor(output.size()).fill_(0)
        for i in range(pt.shape[0]):
            mask[i, 0, pt[i, 1] - 1, pt[i, 0] - 1] = 1
        return Variable(mask.cuda())

    def forward(self, output, ptA, ptB, target_relation):
        # get points here using mask

        # ptA_pred, ptB_pred = self._get_value(output, ptA), self._get_value(output, ptB)
        # target_relation = target_relation[

        # change list of ordinal relations with simbols to numbers +1 -1 0 to use the mask
        mask_A = self._create_position_mask(output, ptA).byte()
        mask_B = self._create_position_mask(output, ptB).byte()

        ptA_pred = output.masked_select(mask_A)
        ptB_pred = output.masked_select(mask_B)

        loss_closer = self._log_loss(ptB_pred, ptA_pred)
        loss_further = self._log_loss(ptA_pred, ptB_pred)
        loss_equal = self._mse(ptA_pred, ptB_pred)

        mask_closer = Variable(target_relation.eq(1).cuda()).float()
        mask_further = Variable(target_relation.eq(-1).cuda()).float()
        mask_equal = Variable(target_relation.eq(0).cuda()).float()

        loss = ((loss_closer * mask_closer) +
                (loss_further * mask_further) +
                (loss_equal * mask_equal))

        return torch.mean(loss)
