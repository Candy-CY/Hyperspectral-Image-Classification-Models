import os
import sys
import time
import glob
import logging
import time
import datetime

import numpy as np
import torch
import darts.utils as utils
import logging
import argparse
import torch.nn as nn
import darts.genotypes as genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
import PIL.Image as Image
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from checkpoint import Checkpointer
from thop import profile	

from torch.autograd import Variable
from darts.model import NetworkHSI as Network
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from data_prepare import readdata
import matplotlib.pyplot as plt
import tensorflow as tf
from loss import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser("HSI")    #15   13   9 
parser.add_argument('--num_class', type=int, default=15, help='classes of HSI dataset')  #每个数据集类别数
                                             #176  144 103 102
parser.add_argument('--band', type=int, default=144, help='spectral of HSI dataset')  #每个数据集光谱
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--infer_batch_size', type=int, default=32, help='infer batch size')
parser.add_argument('--test_batch_size', type=int, default=99, help='infer batch size')
parser.add_argument('--learning_rate', type=float, default=0.004, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')  
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

ktimes=1
t_samples=20
datasetm='houston'  #houston ksc pu pc
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_train_test-'+str(ktimes)+'.txt')  #paviaU
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


def main(genotype, seed):

  data, shuffle_number = readdata(image_file, label_file, datasetm, train_nsamples=t_samples, validation_nsamples=10,
                                  windowsize=27, istraining=True, rand_seed=seed)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)  
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)  


  nBand=args.band   #光谱通道数144-HOUSTON--103--PAVIAU---102--paviaC
  smooth_loss = SMLoss(label_smooth=0.05, class_num=args.num_class)
  smooth_loss=smooth_loss.cuda()
  focal_loss = FocalLoss(gamma=2)
  focal_loss=focal_loss.cuda()
  model = Network(nBand, args.init_channels, args.num_class, args.layers,args.auxiliary, genotype)
  model = model.cuda()
  model.drop_path_prob = args.drop_path_prob
  logging.info('genotype=%s', genotype)
  print('----------------model')
  device = torch.device("cuda")
  input = torch.randn(1,nBand, 27, 27).to(device)
  flops, params = profile(model, inputs=(input,))
  print('flops',flops)			## 打印计算量
  print('params',params)	


  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)
  model.drop_path_prob = args.drop_path_prob

  checkpointer = Checkpointer(
        model, optimizer, scheduler, 'models/'+datasetm+'/', save_to_disk=True)
  writer = SummaryWriter('models/'+datasetm+'/log'+str(ktimes), comment=datasetm)
  arguments = {}
  arguments["epoch"] = 0
  extra_checkpoint_data = checkpointer.load("")
  arguments.update(extra_checkpoint_data)
  checkpointer_period =100
  val_period = 100
  max_epoch = args.epochs
  arch_start_epoch = 20
  start_epoch = arguments["epoch"]
  start_training_time = time.time()
  min_val_obj = 100
  for epoch in range(start_epoch, max_epoch):
    epoch = epoch + 1
    arguments["epoch"] = epoch
    tic = time.time()
    scheduler.step()

    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # training
    train_acc, train_obj, tar, pre = train(data.train, model, focal_loss,smooth_loss,optimizer)
    # validation
    valid_acc, valid_obj, tar_v, pre_v = infer(data.validation, model, focal_loss,smooth_loss)
    toc = time.time()

    logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch + 1,
                train_obj, train_acc, valid_obj, valid_acc, toc - tic)
    writer.add_scalars('acc', { 'val_acc': valid_acc}, epoch)
    writer.add_scalars('loss', {'val_loss': valid_obj}, epoch)
    writer.add_scalars('acc', { 'train_acc': train_acc}, epoch)
    writer.add_scalars('loss', {'train_loss': train_obj}, epoch)
    if epoch > args.epochs * 0.8 and valid_obj < min_val_obj:
      min_val_obj = valid_obj
      checkpointer.save("model_best", **arguments)
    if epoch % checkpointer_period == 0:
        checkpointer.save("model_{:03d}".format(epoch), **arguments)
    if epoch == max_epoch:
        checkpointer.save("model_final", **arguments)

  total_training_time = time.time() - start_training_time
  total_time_str = str(datetime.timedelta(seconds=total_training_time))
  logging.info("Total training time: {}".format(total_time_str))
  model.drop_path_prob = args.drop_path_prob 
  trained_model_dir='models/'+datasetm+'/model_best.pth'
  model_state_dict = torch.load(trained_model_dir).pop("model")
  model.load_state_dict(model_state_dict)

  model.eval()

  y_pred,y_test = test_model(model, shuffle_number, seed)
  return y_pred,y_test





def train(train_data, model, focal_loss,smooth_loss, optimizer):

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.train()
  tar = np.array([])
  pre = np.array([])


  total_batch = int(train_data.num_examples / args.batch_size)
  for i in range(total_batch):
    input, target = train_data.next_batch(args.batch_size)
    input = Variable(torch.from_numpy(input), requires_grad=False).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), requires_grad=False).cuda(async=True)
  

    optimizer.zero_grad()
    logits = model(input)
    loss1 = focal_loss(logits, target)
    loss2 = smooth_loss(logits, target)
    loss=loss1+loss2

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, t, p = utils.accuracy(logits, target, topk=(1,))
    n = input.shape[0]
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

  total_batch = valid_data.num_examples // args.infer_batch_size
  for i in range(total_batch):
    input, target = valid_data.next_batch(args.infer_batch_size)

    input = Variable(torch.from_numpy(input), volatile=True).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)

    logits = model(input)
    loss1 = focal_loss(logits, target)
    loss2 = smooth_loss(logits, target)
    loss=loss1+loss2


    prec1, t, p = utils.accuracy(logits, target, topk=(1,))
    n = input.shape[0]
    objs.update(loss.item(), n)
    top1.update(prec1[0].item(), n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())
  return top1.avg, objs.avg, tar, pre


def test_model(model, shuffle_number, seed):
  model.eval()
  total_tar = np.array([])
  total_pre = np.array([])

  i = 0
  test_nsamples = 0
  time_sum = 0
  while 1:
    data = readdata(image_file, label_file, datasetm,train_nsamples=t_samples, validation_nsamples=10,
                    windowsize=27, istraining=False, shuffle_number=shuffle_number, times=i, rand_seed=seed)
    if data == None:
      print('test_nsamples:',test_nsamples)
      logging.info('time_cost_all:{} '.format(time_sum))
      print('done')
      return total_tar, total_pre

    test_nsamples += data.test.num_examples

    x_test, y_test = data.test.images, data.test.labels
    add_samples = args.test_batch_size - data.test.num_examples % args.test_batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)

    tar = np.array([])
    pre = np.array([])
    total_batch = (data.test.num_examples + add_samples) // args.test_batch_size
    for j in range(total_batch):
      input, target = x_test[j*args.test_batch_size:(j+1)*args.test_batch_size, :, :, :], y_test[j*args.test_batch_size:(j+1)*args.test_batch_size, :]
      input = Variable(torch.from_numpy(input), volatile=True).cuda()
      target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)
      time_s = time.time()
      logits = model(input)
      time_e = time.time()
      time_sum += (time_e - time_s)

      _, t, p = utils.accuracy(logits, target, topk=(1,))
      # print(p)
      tar = np.append(tar, t.data.cpu().numpy())
      pre = np.append(pre, p.data.cpu().numpy())


    total_tar = np.append(total_tar, tar[add_samples:])
    total_pre = np.append(total_pre, pre[add_samples:])
    i = i + 1


def cal_results(matrix):         #计算精度
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA




def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (Y_pred,y_test,name):
  if name == 'houston':
    target_names = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15']
  if name == 'ksc':
    target_names = ['1', '2','3','4','5','6','7','8','9','10','11','12','13']
  if name == 'pu':
    target_names = ['1','2','3','4', '5','6','7','8 ','9']
  if name == 'pc':
    target_names = ['1','2','3','4', '5','6','7','8 ','9']
  
  oa = accuracy_score(y_test, y_pred)
  confusion = confusion_matrix(y_test, y_pred)
  each_acc, aa = AA_andEachClassAccuracy(confusion)
  kappa = cohen_kappa_score(y_test, y_pred)  
  return confusion, oa*100, each_acc*100, aa*100, kappa*100



if __name__ == '__main__':
  torch.set_num_threads(1)
  genotype = eval('genotypes.{}'.format(args.arch))
  y_pred,y_test= main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1)) 

  confusion,  oa, each_acc, aa, kappa = reports(y_pred,y_test,datasetm)
  confusion = str(confusion)
  file_name = "map/classification_report_"+datasetm+"-"+str(ktimes)+".txt"
  with open(file_name, 'w') as x_file:
      x_file.write(' Overall accuracy (%):{}'.format(oa))
      x_file.write('\n')
      x_file.write(' Average accuracy (%):{}'.format(aa))
      x_file.write('\n')
      x_file.write(' Kappa accuracy (%):{}'.format(kappa))
      x_file.write('\n')
      for i in range(len(each_acc)):
            x_file.write('class {} acc: {}\n'.format(i+1, each_acc[i]))
      x_file.write('\n')
      x_file.write('{}'.format(confusion))

