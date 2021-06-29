import sys, os
sys.path.append(os.getcwd())

import numpy as np
import visdom
from ipdb import set_trace as st
from scipy.misc import imresize
# Based on visualizer.py from pix2pix pytorch
# Save errors in chechpoint folder

# mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225]

# mean=(0.5, 0.5, 0.5)
# std=(0.5, 0.5, 0.5)

input_shape = (3, 256, 256)
value = 255.0


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.name = opt.name
        self.message = opt.name
        self.graph_started = False
        self.opt = opt

        # if opt.dataset_name == 'kitti':
        #     global value = 1/256

        if self.display_id > 0:
            self.vis = visdom.Visdom(port=opt.port)
        # Here, p2p saves logfile

    def tensor2im(self, img, imtype=np.uint8, convert_value=255.0):
        # ToDo: improve this horrible function
        if img.shape[1] > 3:   # case of focalstack
            img = img[:, :3]
        
        if(type(img) != np.ndarray):
            image_numpy = img[0].cpu().float().numpy()
        else:
            image_numpy = img
        if img.shape[1] == 3:
            image_numpy = (image_numpy + 1) / 2.0 * convert_value
            image_numpy = image_numpy.astype(imtype)
        else:
            image_numpy = (image_numpy - image_numpy.min()) * (255 / self.opt.max_distance)
            image_numpy = np.repeat(image_numpy, 3, axis=0)
        return image_numpy

    # visuals: dictionary of images to display or save
    def display_images(self, visuals, epoch, table=True, phase='train'):
        idx = self._get_display_id(phase)
        if self.display_id > 0:
            if table:
                for i, (label, image_numpy) in enumerate(visuals.items()):
                    if i == 0:
                        image_conc = self.tensor2im(image_numpy)
                        label_conc = label
                    else:
                        if 'sem' in label:
                            from .util import labels_to_colors
                            image = labels_to_colors(image_numpy, self.opt.color_palette).astype(np.uint8).transpose([2,0,1])

                            image_conc = np.concatenate((image_conc, image), axis=1)
                            label_conc += ('\t' + label)
                        else:
                            image = self.tensor2im(image_numpy)
                            image_conc = np.concatenate((image_conc, image), axis=1)
                            label_conc += ('\t' + label)

                self.vis.image(image_conc,
                               opts=dict(title='{} Epoch[{}] '.format(self.name, epoch) + label_conc), win=self.display_id + idx)

            else:
                st()
                for label, image_numpy in visuals.items():
                    self.vis.image((self.tensor2im(image_numpy)), opts=dict(title='{} Epoch[{}] '.format(self.name, epoch) + label), win=self.display_id + idx)

                    idx += 1

    def display_errors(self, errors, epoch, counter_ratio, phase='train'):
        if self.display_id > 0:
            self._create_plot_data(phase, errors)    # if non existing
            plot_data = self.get_plot_data(phase)
            plot_data['X'].append(epoch + counter_ratio)
            plot_data['Y'].append([errors[k] for k in plot_data['legend']])
            self.vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': self._get_title(phase),
                    'legend': plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self._get_display_id(phase))

    def display_existing_plot(self, plot_data, phase):
        self.vis.line(
                X = np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y = np.array(plot_data['Y']),
                opts={
                    'title': self._get_title(phase),
                    'legend': plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self._get_display_id(phase))

    def print_errors(self, errors, epoch, i, len_loader, t):
        total_epochs = self.opt.nEpochs
        self.message = '===> Epoch[{}/{}]({}/{})'.format(epoch, total_epochs, i, len_loader)
        for k, v in errors.items():
            self.message += ' {}: {:.4f}'.format(k, v)
        self.message
        # print(self.message)
        return self.message

    def save_errors_file(self, logfile):
        logfile.write(self.message + '\n')

    def save_errors(self):
        print('to implement. Checkpoints are on opt')

    def _get_display_id(self, phase):
        # changes if validation, or loss
        if phase == 'train':
            return self.display_id
        else:
            return self.display_id + 20

    def _get_title(self, phase):
        if phase == 'train':
            return self.name + ' loss over time'
        else:
            return self.name + ' validation over time'

    def _create_plot_data(self, phase, errors):
        if phase == 'train':
            if not hasattr(self, 'plot_data'):
                self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys()), 'color': 'red'}
        else:
            if not hasattr(self, 'plot_data_val'):
                self.plot_data_val = {'X': [], 'Y': [], 'legend': list(errors.keys()), 'color': 'green'}

    def get_plot_data(self, phase):
        if phase == 'train':
            return self.plot_data
        else:
            return self.plot_data_val
            