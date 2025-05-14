from HyperTools import *
import argparse
import os

def main(args):  
    dataID = args.dataID
    X,Y,train_array,test_array = LoadHSI(dataID,args.train_samples)

    if dataID==1:
        save_pre_dir = './Data/PaviaU/'
    elif dataID==2:
        save_pre_dir = './Data/Salinas/'
    elif dataID==3:
        save_pre_dir = './Data/IP/'
    elif dataID==4:
        save_pre_dir = './Data/KSC/'
    elif dataID==5:
        save_pre_dir = './Data/xuzhou/'
    elif dataID==6:
        save_pre_dir = './Data/Houston/'

    Y -= 1

    if os.path.exists(save_pre_dir)==False:
        os.makedirs(save_pre_dir)
    # np.load和np.save是读写磁盘数组数据的两个主要函数，
    # 默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中。
    # 他们会自动处理元素类型和形状等信息，“.npy”格式将数组保存到二进制文件中。
    
    np.save(save_pre_dir+'X.npy',X)
    np.save(save_pre_dir+'Y.npy',Y)
    np.save(save_pre_dir+'train_array.npy',train_array)
    np.save(save_pre_dir+'test_array.npy',test_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--train_samples', type=int, default=300)

    main(parser.parse_args())