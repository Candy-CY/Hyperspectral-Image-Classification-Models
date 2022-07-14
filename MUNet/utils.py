import torch
import torch.nn as nn
import numpy as np

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

def l12_norm(inputs):
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out

class MinVolumn(nn.Module):
    def __init__(self, band, num_classes, delta):
        super(MinVolumn, self).__init__()
        self.band = band
        self.delta = delta
        self.num_classes = num_classes
    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band,self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss

class SparseLoss(nn.Module):
    def __init__(self, sparse_decay):
        super(SparseLoss, self).__init__()
        self.sparse_decay = sparse_decay

    def __call__(self, input):
        loss = l12_norm(input)
        return self.sparse_decay*loss

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1)
