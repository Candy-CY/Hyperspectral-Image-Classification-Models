# -*- coding: utf-8 -*-
"""
training configuration
"""

import argparse
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### common parameters
parser.add_argument('--scale_factor',type=int,default=5, help='scale_factor, set to 5 and 8 in our experiment')
parser.add_argument('--sp_root_path',type=str, default='data/spectral_response/',help='where you store your own spectral response')
parser.add_argument('--default_datapath',type=str, default="data/",help='where you store your HSI data file and spectral response file')

parser.add_argument('--data_name',type=str, default="houston18",help='name of your data and spectral response. houston18 is given as a example')
parser.add_argument('--isCal_SRF',type=str, default="Yes",help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=str, default="Yes",help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')
parser.add_argument('--batchsize',type=int, default=1,help='')
parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids: e.g. 0;1;2')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoints',help='where store the training results')
parser.add_argument('--hsi_channel',type=int,default=46,help='the channels of hyperspectral image')
parser.add_argument('--msi_channel',type=int,default=8,help='the channels of multispectral image')
parser.add_argument('--image_size',type=int,default=240,help='the size of the image')

#the first stage
parser.add_argument("--lr_stage1", type=float, default=8e-4,help='learning rate')
parser.add_argument("--epoch_stage1", type=int, default=40000, help='total epoch')
parser.add_argument("--decay_begin_epoch_stage1", type=int, default=10000, help='epoch which begins to decay,so the lr is 1e-3 in the first 10000 epochs and then it decays from 10000th epoch to 20000th epoch. When 20000 epochs are finished, the lr decays to 0')

#the second stage
parser.add_argument("--lr_stage2", type=float, default=1e-3 + 3e-4,help='learning rate')
parser.add_argument("--epoch_stage2", type=int, default=200000, help='total epoch')
parser.add_argument("--decay_begin_epoch_stage2", type=int, default=20000, help='epoch which begins to decay,so the lr is 1e-3 in the first 20000 epochs and then it decays from 20000th epoch to 30000th epoch. When 30000 epochs are finished, the lr decays to 0')

# augmenentor
parser.add_argument('--learning_rate_a', default=1e-4, type=float, help='learning rate in training')
parser.add_argument('--decay_rate', type=float, default=1e-16, help='decay rate of learning rate')
# parser.add_argument("--decay_begin_augmentor",type=1000,help="dasidas")

args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu')
args.device=device
# Because the full width at half maxima of Gaussian function used to generate the PSF is set to scale factor in our experiment,
# there exists the following relationship between  the standard deviation and scale_factor :
args.sigma = args.scale_factor / 2.35482
