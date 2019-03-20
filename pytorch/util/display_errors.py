# loads dicts and displays errors by the name - use base model!
import os
import _pickle as pickle
# from collections import OrderedDict
from visualizer import Visualizer
import argparse
import glob
#from ipdb import set_trace as st

# usage: python util/display_errors.py --port 8009 --server all --val --train

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='')
parser.add_argument('--port', default=8001)
parser.add_argument('--display_id', type=int, default=1)
parser.add_argument('--checkpoints_path', default='./checkpoints')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--server', default='all')
parser.add_argument('--s_string', default='')
opt = parser.parse_args()


def _get_plot_data_filename(opt, phase):
    dirpath = opt.name
    return os.path.join(dirpath,
                        'plot_data' + ('' if phase == 'train' else '_' + phase) + '.p')


def _load_plot_data(filename):
    # verify if file exists
    # if not os.path.isfile(filename):
    #     raise Exception('In _load_plot_data File {} doesnt exist.'.format(filename))
    # else:
    if os.path.isfile(filename):
        return pickle.load(open(filename, "rb"))
    else:
        return None


def load_plot_data(opt):
    visualizer.plot_data = _load_plot_data(_get_plot_data_filename(opt, 'train'))
    is_load_data = visualizer.plot_data is not None
    visualizer.plot_data_val = _load_plot_data(_get_plot_data_filename(opt, 'val'))
    is_load_data_val = visualizer.plot_data_val is not None
    return is_load_data, is_load_data_val

# In madmax8
# if opt.server == 'mad8':
#     project_list = (
#                 'nyu_sornet_resnet_decoder_masked_pretrained_L1_12k',
#                 'nyu_resnet50_sornet_pt_m_L2_12k',
#                 'nyu_sornet_resnet_decoder_masked_pretrained_L1_230k',
#                 'nyu_sornet_resnet_decoder_masked_pt_L1_12k',
#                 'train_nyu_resnet152_sornet_m_pt_L1_230',
#                 'nyu_resnet50_sornet_pt_m_berhu_deltaLaina_12k',
# )
# el


if opt.name == '':
    # project_list = sorted(os.listdir(opt.checkpoints_path))
    project_list = sorted(glob.glob(os.path.join(opt.checkpoints_path, '*{}*'.format(opt.s_string))))
else:
    project_list = [opt.name]

for opt.name in project_list:
    visualizer = Visualizer(opt)
    is_load_data, is_load_data_val = load_plot_data(opt)
    if opt.train and is_load_data:
        visualizer.display_existing_plot(visualizer.plot_data, 'train')
    if opt.validate and is_load_data_val:
        visualizer.display_existing_plot(visualizer.plot_data_val, 'val')
    if is_load_data or is_load_data_val:
        print('Data from {}'.format(os.path.basename(opt.name)))
    #if not is_load_data and not is_load_data_val:
    #    print('NO Data from {}'.format(opt.name))
    opt.display_id += 20
