
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


class SMLoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=16):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
      ''' 
      Args:
      pred: prediction of model output    [N, M]
      target: ground truth of sampler [N]
      '''
      eps = 1e-12
      
      if self.label_smooth is not None:
        # cross entropy loss with label smoothing
        logprobs = F.log_softmax(pred, dim=1)	# softmax + log
        # print(logprobs.shape,target.shape)
        
        target = one_hot(target,self.class_num)
        logprobs= logprobs.cuda()
        target= target.cuda()

        
        target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
        loss = -1*torch.sum(target*logprobs, 1)
        
        
      else:
        # standard cross entropy loss
        loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

      return loss.mean()
