import os
import sys
import time
import glob
import numpy as np
import torch
import darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.autograd import Variable
from darts.model_search import Network
from darts.architect import Architect
from sklearn.metrics import confusion_matrix
from data_prepare import readdata
import random
from loss import *
import torchvision
import torch.nn.functional as F
from IPython.display import display

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=15, help='classes of HSI dataset')  #9,15,16,13每个数据集的类别
parser.add_argument('--band', type=int, default=144, help='spectral of HSI dataset')  #每个数据集光谱
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=4e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
args = parser.parse_args()

datasetm='houston'  #houston ksc pu pc
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_search.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if datasetm=='houston':
  image_file = r'data/houston/houston.mat'   
  label_file = r'data/houston/houston_gt_sum.mat'   
if datasetm=='ksc':
  image_file = r'data/ksc/KSC.mat'   
  label_file = r'data/ksc/KSC_gt.mat'   
if datasetm=='pu':
  image_file = r'data/paviaU/PaviaU.mat'  
  label_file = r'data/paviaU/PaviaU_gt.mat'   
if datasetm=='pc':
  image_file = r'data/paviaC/Pavia.mat'   
  label_file = r'data/paviaC/Pavia_gt.mat'   




def main(seed):

  data, shuffle_number = readdata(image_file, label_file, datasetm, train_nsamples=20, validation_nsamples=10,
                                  windowsize=27, istraining=True, rand_seed=seed)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(seed)


  nBand=args.band  
  smooth_loss = SMLoss(label_smooth=0.05, class_num=args.num_class)
  smooth_loss=smooth_loss.cuda()
  focal_loss = FocalLoss(gamma=2)
  focal_loss=focal_loss.cuda()
  model = Network(nBand,args.init_channels, args.num_class, args.layers, focal_loss,smooth_loss)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.25)
  architect = Architect(model, args)
  min_valid_obj = 100

  genotype = model.genotype()
  print('genotype = ', genotype)

  for epoch in range(args.epochs):
    tic = time.time()
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %03d lr %e', epoch+1, lr)
    # training
    train_acc, train_obj, tar, pre = train(data.train, data.validation, model, architect, focal_loss,smooth_loss, optimizer, lr)
    # validation
    valid_acc, valid_obj, tar_v, pre_v = infer(data.validation, model, focal_loss,smooth_loss)
    toc = time.time()

    logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch + 1, train_obj, train_acc, valid_obj, valid_acc, toc-tic)

    if valid_obj < min_valid_obj:
      genotype = model.genotype()
      logging.info('genotype=%s', genotype)
      print('genotype = ', genotype)
      min_valid_obj = valid_obj

  return genotype


def train(train_data, valid_data, model, architect, focal_loss,smooth_loss, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  tar = np.array([])
  pre = np.array([])

  total_batch = int(train_data.num_examples / args.batch_size)
  for i in range(total_batch):
    input, target = train_data.next_batch(args.batch_size)

    model.train()
    n = input.shape[0]

    input = Variable(torch.from_numpy(input), requires_grad=False).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), requires_grad=False).cuda(async=True)

    input_search, target_search = valid_data.next_batch(args.batch_size)
    input_search = Variable(torch.from_numpy(input_search), requires_grad=False).cuda()
    target_search = Variable(torch.from_numpy(np.argmax(target_search, axis=1)), requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss1 = focal_loss(logits, target)
    loss2 = smooth_loss(logits, target)
    loss=loss1+loss2

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, t, p = utils.accuracy(logits, target, topk=(1, ))
    objs.update(loss.item(), n)
    top1.update(prec1[0].item(), n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre


def infer(valid_data, model, focal_loss,smooth_loss):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  tar = np.array([])
  pre = np.array([])

  total_batch = valid_data.num_examples // args.batch_size
  for i in range(total_batch):
    input, target = valid_data.next_batch(args.batch_size)
    n = input.shape[0]

    input = Variable(torch.from_numpy(input), volatile=True).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)

    logits = model(input)
    loss1 = focal_loss(logits, target)  
    loss2 = smooth_loss(logits, target)
    loss=loss1+loss2

    prec1, t, p = utils.accuracy(logits, target, topk=(1, ))
    objs.update(loss.item(), n)
    top1.update(prec1[0].item(), n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre




if __name__ == '__main__':
  genotype = main(seed=np.random.randint(low=0, high=10000, size=1))
  print('Searched Neural Architecture:')
  print(genotype)

