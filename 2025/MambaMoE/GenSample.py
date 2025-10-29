import numpy as np

from HyperTools import *
import argparse
import os

def main(args):  
    dataID = args.dataID
    X, Y, train_array, test_array = LoadHSI(dataID, args.train_num)

    if dataID==1:
        save_pre_dir = './Data/PaviaU_15/'
    elif dataID==3:
        save_pre_dir = './Data/Houston_15/'
    elif dataID==6:
        save_pre_dir = './Data/Whuhi_hanchuan_30/'
   
    Y -= 1

    if os.path.exists(save_pre_dir)==False:
        os.makedirs(save_pre_dir)

    np.save(save_pre_dir+'X.npy',X)
    np.save(save_pre_dir+'Y.npy',Y)
    np.save(save_pre_dir+'train_array.npy',train_array)
    np.save(save_pre_dir+'test_array.npy',test_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--train_num', type=int, default=10)
    parser.add_argument('--val_num', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./')

    main(parser.parse_args())