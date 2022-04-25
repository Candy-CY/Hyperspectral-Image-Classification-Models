import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def cal_results(matrix):
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


def predVisIN(indices, pred, size1, size2):
    
    if pred.ndim > 1:
        pred = np.ravel(pred)
    
    x = np.zeros(size1*size2)
    x[indices] = pred
    
    y = np.ones((x.shape[0], 3))

    for index, item in enumerate(x):
        if item == 0:
            y[index] = np.array([230, 230, 230]) / 255.  # np.array([255, 255, 255]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 7:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 9:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 10:  
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 12:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 14:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 15:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 16:
            y[index] = np.array([255, 215, 0]) / 255.
    
    y_rgb = np.reshape(y, (size1, size2, 3))
    
    return y_rgb

def visPC(indices, pred, size1, size2):
    
    if pred.ndim > 1:
        pred = np.ravel(pred)
    
    x = np.zeros(size1*size2)
    x[indices] = pred[indices]
    
    y = np.ones((x.shape[0], 3))

    for index, item in enumerate(x):
        if item == 0:
            y[index] = np.array([230, 230, 230]) / 255.  #np.array([255, 255, 255]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 7:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 9:
            y[index] = np.array([128, 0, 0]) / 255.
    
    y_rgb = np.reshape(y, (size1, size2, 3))
    
    return y_rgb

