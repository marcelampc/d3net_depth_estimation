import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt
from tqdm import tqdm
from .train_model import TrainModel
from networks import networks

import util.pytorch_ssim as pytorch_ssim

from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import numpy as np

# Be able to add many loss functions
class MultiTaskGen(TrainModel):
    def name(self):
        return 'MultiTask General Model'

    def initialize(self, opt):
        TrainModel.initialize(self, opt)

        if self.opt.resume:
            self.netG, self.optimG = self.load_network()
        elif self.opt.train:
            from os.path import isdir
            if isdir(self.opt.pretrained_path) and self.opt.pretrained:
                self.netG = self.load_weights_from_pretrained_model()
            else:
                self.netG = self.create_network()
            self.optimG = self.get_optimizerG(self.netG, self.opt.lr,
                                              weight_decay=self.opt.weightDecay)
            # self.criterion = self.create_reg_criterion()
        self.n_tasks = len(self.opt.tasks)

        if self.opt.display_id > 0:
            self.errors = OrderedDict()
            self.current_visuals = OrderedDict()
        if 'depth' in self.opt.tasks:
            self.criterion_reg = self.get_regression_criterion()
        if 'semantics' in self.opt.tasks:
            self.initialize_semantics()
        if 'instance' in self.opt.tasks:
            pass
        if 'normals' in self.opt.tasks:
            pass

    def initialize_semantics(self):
        from util.util import get_color_palette, get_dataset_semantic_weights
        self.global_cm = np.zeros((self.opt.n_classes-1, self.opt.n_classes-1))
        self.target = self.get_variable(torch.LongTensor(self.batchSize, self.opt.output_nc, self.opt.imageSize[0], self.opt.imageSize[1]))
        self.outG_np = None
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0
        self.opt.color_palette = get_color_palette(self.opt.dataset_name)

        weights = self.get_variable(torch.FloatTensor(get_dataset_semantic_weights(self.opt.dataset_name)))
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
    
    def train_batch(self):
        self._train_batch()

    def restart_variables(self):
        self.it = 0
        self.n_iterations = 0
        self.n_images = 0
        self.rmse = 0
        self.e_reg = 0
        self.norm_grad_sum = 0

    def mean_errors(self):
        if 'depth' in self.opt.tasks:
            rmse_epoch = self.rmse / self.n_images
            self.set_current_errors(RMSE=rmse_epoch)

    def get_errors_regression(self, target, output):
        if self.total_iter % self.opt.print_freq == 0:

            # gets valid pixels of output and target
            if not self.opt.no_mask:
                (output, target), n_valid_pixls = self.apply_valid_pixels_mask(output, target, value=self.opt.mask_thres)

            with torch.no_grad():
                e_regression = self.criterion_reg(output, target.detach())
                for k in range(output.shape[0]):
                    self.rmse += sqrt(self.mse_scaled_error(output[k], target[k], n_valid_pixls).item()) # mean through the batch
                    self.n_images += 1

            self.set_current_visuals(depth_gt=target.data,
                                        depth_out=output.data)
            self.set_current_errors(L1=e_regression.item())
            
            # return e_regression

    def get_errors_semantics(self, target, output, n_classes):
        # e_semantics = self.cross_entropy(output, target)
        if self.total_iter % self.opt.print_freq == 0:
            with torch.no_grad():
                target_sem_np = target.cpu().numpy()
                output_np = np.argmax(output.cpu().data.numpy(), axis=1)
                cm = confusion_matrix(target_sem_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
                self.global_cm += cm[1:,1:]

                # scores
                overall_acc = metrics.stats_overall_accuracy(self.global_cm)
                average_acc, _ = metrics.stats_accuracy_per_class(self.global_cm)
                average_iou, _ = metrics.stats_iou_per_class(self.global_cm)

                self.set_current_errors(OAcc=overall_acc, AAcc=average_acc, AIoU=average_iou)
                self.set_current_visuals(sem_gt=target.data[0].cpu().float().numpy(),
                                        sem_out=output_np[0])

            # return e_semantics

    def get_errors_instance(self, target, output):
        pass

    def get_errors_normals(self, target, output):
        pass

    def _train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda)
        input_data.requires_grad = True
        self.set_current_visuals(input=input_data.data)
        batch_size = input_cpu.shape[0]

        outG = self.netG.forward(input_data)
        
        losses = []
        # norm_grad = []
        for i_task, task in enumerate(self.opt.tasks):
            target = target_cpu[i_task].to(self.cuda)
            if task == 'semantics':
                losses.append(self.cross_entropy(outG[i_task], target))
                self.get_errors_semantics(target, outG[i_task], n_classes=self.opt.outputs_nc[i_task])
            elif task == 'depth':
                losses.append(self.criterion_reg(target, outG[i_task]))
                self.get_errors_regression(target, outG[i_task])

        self.loss_error = sum(losses)

        self.optimG.zero_grad()
        self.loss_error.backward()
        self.optimG.step()

        self.n_iterations += 1 # outG[0].shape[0]

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            # self.norm_grad_sum += np.array(norm_grad).mean()
            # norm_grad = self.norm_grad_sum / self.n_iterations
            # self.set_current_errors(norm_grad=norm_grad)

    def evaluate(self, data_loader, epoch):
        if self.opt.validate and self.total_iter % self.opt.val_freq == 0:
            self.get_eval_error(data_loader)
            self.visualizer.display_errors(self.val_errors, epoch, float(self.it)/self.len_data_loader, phase='val')
            message = self.visualizer.print_errors(self.val_errors, epoch, self.it, len(data_loader), 0)
            print('[Validation] ' + message)
            self.visualizer.display_images(self.val_current_visuals, epoch=epoch, phase='val')

    def get_eval_error(self, data_loader):
        model = self.netG.train(False)
        self.val_errors = OrderedDict()
        self.val_current_visuals = OrderedDict()

        losses = np.zeros(self.n_tasks)
        with torch.no_grad():
            pbar_val = tqdm(range(len(data_loader)))
            data_iter = iter(data_loader)
            for _ in pbar_val:
                pbar_val.set_description('[Validation]')
                input_cpu, target_cpu = data_iter.next()
                input_data = input_cpu.to(self.cuda)
                
                outG = model.forward(input_data)
                
                for i_task, task in enumerate(self.opt.tasks):
                    target = target_cpu[i_task].to(self.cuda)
                    if task == 'semantics':
                        target_np = target_cpu[i_task].data.numpy()
                        output_np = np.argmax(outG[i_task].cpu().data.numpy(), axis=1)
                        cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(self.opt.outputs_nc[i_task])))
                        target_cpu[i_task] = target.data[0].cpu().float().numpy()
                        outG[i_task] = output_np[0]
                        loss, _ = metrics.stats_iou_per_class(cm[1:,1:])
                        losses[i_task] += loss
                    elif task == 'depth':
                        losses[i_task] += sqrt(nn.MSELoss()(target, outG[i_task]))
                        outG[i_task] = outG[i_task].data

            self.val_current_visuals.update([('input', input_cpu)])
            for i_task, task in enumerate(self.opt.tasks):
                self.val_errors.update([('l_{}'.format(task), losses[i_task]/len(data_loader))])
    
                self.val_current_visuals.update([('t_{}'.format(task), target_cpu[i_task])])
                self.val_current_visuals.update([('o_{}'.format(task), outG[i_task])])

    def set_current_errors_string(self, key, value):
        self.errors.update([(key, value)])

    def set_current_errors(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.errors.update([(key, value)])

    def get_current_errors(self):
        return self.errors

    def get_current_errors_display(self):
        return self.errors

    def set_current_visuals(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.current_visuals.update([(key, value)])

    def get_current_visuals(self):
        return self.current_visuals

    def get_checkpoint(self, epoch):
        return ({'epoch': epoch,
                 'arch_netG': self.opt.net_architecture,
                 'state_dictG': self.netG.state_dict(),
                 'optimizerG': self.optimG,
                 'best_pred': self.best_val_error,
                 'tasks': self.opt.tasks,
                 'mtl_method': self.opt.mtl_method,
                #  'data_augmentation': self.opt.data_augmentation, # used before loading net
                 'n_classes': self.opt.n_classes,
                 })

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            self.start_epoch = checkpoint['epoch']
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            netG.load_state_dict(checkpoint['state_dictG'])
            optimG = checkpoint['optimizerG']
            self.best_val_error = checkpoint['best_pred']
            self.print_save_options()
            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG, optimG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

    def load_weights_from_pretrained_model(self):
        epoch = 'best'
        checkpoint_file = os.path.join(self.opt.pretrained_path, epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(epoch, self.opt.pretrained_path))
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            model_dict = netG.state_dict()
            pretrained_dict = checkpoint['state_dictG']
            model_shapes = [v.shape for k, v in model_dict.items()]
            exclude_model_dict = [k for k, v in pretrained_dict.items() if v.shape not in model_shapes]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in exclude_model_dict}
            model_dict.update(pretrained_dict)
            netG.load_state_dict(model_dict)
            _epoch = checkpoint['epoch']
            # netG.load_state_dict(checkpoint['state_dictG'])
            print("Loaded model from epoch {}".format(_epoch))
            return netG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.pretrained_path + '/' + epoch))

    def to_numpy(self, data):
        return data.data.cpu().numpy()