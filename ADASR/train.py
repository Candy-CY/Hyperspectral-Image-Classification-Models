import time

import torch
import numpy as np
import random
from utils.func import print_options
from arch.spectral_upsample import Spectral_upsample
from config import args
from Data_loader import Dataset
from model import sr_model

# 设置固定的输入值
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    Spectral_up_net = Spectral_upsample(args, args.msi_channel, args.hsi_channel, init_type='normal', init_gain=0.02,
                                             initializer=False)
    #store the training configuration in opt.txt
    #setting
    setup_seed(2)     #seed is set to 2
    print_options(args)
    train_dataset=Dataset(args)
    down_model = sr_model.DownModel(train_dataset)

    # stage one train
    down_model()
    from utils.func import save_net
    save_net(args,down_model.Spectral_down_net)
    save_net(args,down_model.Spatial_down_net)
    # begin stage 2
    up_model = sr_model.UpModel(Spectral_up_net, down_model)
    # stage two train
    up_model()
    from utils.func import save_hhsi
    #from utils.func import save_net
    ##save trained three module
    save_net(args,up_model.Spectral_up_net)

    est_hhsi = up_model.Spectral_up_net(train_dataset[0]["hmsi"].unsqueeze(0).to(args.device))

    ###save estimated HHSI
    save_hhsi(args,est_hhsi)

    print(args)
    print('all done')
    print("end")
