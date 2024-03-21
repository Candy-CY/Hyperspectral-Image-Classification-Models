import os, sys
import json, time
import numpy as np
from torchvision import transforms 
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def show_img(x):
    def trans(x):
        if type(x) == np.ndarray:
            return np.transpose(x, (1, 2, 0))
        else:
            return x.permute(1, 2, 0)

    def totype(x):
        if type(x) == np.ndarray:
            return x.astype(np.uint8)
        else:
            return x.numpy().astype(np.uint8)
    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: trans(t)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: totype(t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(x))

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

    def get_avg(self):
        return self.avg
    
class Recoder(object):
    def __init__(self) -> None:
        self.record_data = {}

    def append_index_value(self, name, index, value):
        """
        index : int, 
        value: Any
        save to dict
        {index: list, value: list}
        """
        if name not in self.record_data:
            self.record_data[name] = {
                "type": "index_value",
                "index":[],
                "value":[]
            } 
        self.record_data[name]['index'].append(index)
        self.record_data[name]['value'].append(value)

    def record_param(self, param):
        self.record_data['param'] = param 

    def record_eval(self, eval_obj):
        self.record_data['eval'] = eval_obj
        
    def to_file(self, path):
        time_stamp = int(time.time())
        save_path = "%s_%s.json" % (path, str(time_stamp))
        ss = json.dumps(self.record_data, indent=4)
        with open(save_path, 'w') as fout:
            fout.write(ss)
            fout.flush()

    def reset(self):
        self.record_data = {}
        

# global recorder
recorder = Recoder()
