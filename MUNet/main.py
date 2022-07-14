import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation import compute_rmse, compute_sad
from utils import print_args, SparseLoss, NonZeroClipper, MinVolumn
from data_loader import set_loader
from model import Init_Weights, MUNet
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import argparse
import random
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--fix_random', action='store_true', help='fix randomness')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu_id', default='0,1,2', help='gpu id')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch', default=1, type=int, help='input data size')
parser.add_argument('--learning_rate_en', default=3e-4, type=float, help='learning rate of encoder')
parser.add_argument('--learning_rate_de', default=1e-4, type=float, help='learning rate of decoder')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='network parameter regularization')
parser.add_argument('--lamda', default=0, type=float, help='sparse regularization')
parser.add_argument('--reduction', default=2, type=int, help='squeeze reduction')
parser.add_argument('--delta', default=0, type=float, help='delta coefficient')
parser.add_argument('--gamma', default=0.8, type=float, help='learning rate decay')
parser.add_argument('--epoch', default=200, type=int, help='number of epoch')
parser.add_argument('--dataset', choices=['muffle','houston170'], default='muffle', help='dataset to use')
args = parser.parse_args()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.cuda.set_device(2)

    if torch.cuda.is_available():
        print ('GPU is true')
        print('cuda version: {}'.format(torch.version.cuda))
    else:
        print('CPU is true')

    if args.fix_random:
        # init seed within each thread
        manualSeed = args.seed
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        # NOTE: literally you should uncomment the following, but slower
        cudnn.deterministic = True
        cudnn.benchmark = False
        print('Warning: You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')
    else:
        cudnn.benchmark = True
    print("Using GPU: {}".format(args.gpu_id))

    # create dataset and model
    train_loaders, test_loaders, label, M_init, M_true, num_classes, band, col, row, ldr_dim = set_loader(args)
    net = MUNet(band, num_classes, ldr_dim, args.reduction).cuda()
    
    # initialize net parameters and endmembers
    if args.dataset == 'muffle':
        position = np.array([0,2,1,3,4]) # muffle
        Init_Weights(net,'xavier', 1)
    elif args.dataset == 'houston170': 
        position = np.array([0,1,2,3]) # houston170
        Init_Weights(net,'xavier', 1)

    net_dict = net.state_dict()
    net_dict['decoder.0.weight'] = M_init
    net.load_state_dict(net_dict)

    # loss funtion and regularization
    apply_nonegative = NonZeroClipper()
    loss_func = nn.MSELoss()
    criterionSparse = SparseLoss(args.lamda)
    criterionVolumn = MinVolumn(band, num_classes, args.delta)

    # optimizer setting
    params = map(id, net.decoder.parameters())
    ignored_params = list(set(params))      
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters()) 
    optimizer = torch.optim.Adam([{'params': base_params},{'params': net.decoder.parameters(), 'lr': args.learning_rate_de}],
                                    lr = args.learning_rate_en, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)

    time_start = time.time()
    for epoch in range(args.epoch):
        for i, traindata in enumerate(train_loaders):        
            net.train()

            x, y = traindata       
            x = x.cuda()
            y = y.cuda()
            
            abu, output = net(x,y)            
            output = torch.reshape(output, (output.shape[0], band))
            x = torch.reshape(x, (output.shape[0], band))

            # reconstruction loss
            MSE_loss = torch.mean(torch.acos(torch.sum(x * output, dim=1)/
                        (torch.norm(output, dim=1, p=2)*torch.norm(x, dim=1, p=2))))
            # sparsity and minimum volume regularization
            MSE_loss += criterionSparse(abu) + criterionVolumn(net.decoder[0].weight)       

            optimizer.zero_grad()
            MSE_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1) 
            optimizer.step()
            net.decoder.apply(apply_nonegative)    
        
        if epoch % 1 == 0:
            print('Epoch: {:d} | Train Unmix Loss: {:.5f} | RE Loss: {:.5f} | Sparsity Loss: {:.5f} | Minvol: {:.5f}'
                .format(epoch, MSE_loss, loss_func(output, x), criterionSparse(abu), criterionVolumn(net.decoder[0].weight)))
            net.eval()
            for k, testdata in enumerate(test_loaders):
                x, y = testdata
                x = x.cuda()
                y = y.cuda()
            
                abu_est, output = net(x, y)
 
            abu_est = torch.reshape(abu_est.squeeze(-1).permute(2,1,0), (num_classes,row,col)).permute(0,2,1).cpu().data.numpy()
            edm_result = torch.reshape(net.decoder[0].weight, (band,num_classes)).cpu().data.numpy()           
            print('RMSE: {:.5f} | SAD: {:.5f}'.format(compute_rmse(abu_est[position,:,:],label), compute_sad(M_true, edm_result[:,position])))
            print('**********************************')

        scheduler.step()
    time_end = time.time()
    # model evaluation
    net.eval()
    print(net.spectral_se)
    for i, testdata in enumerate(test_loaders):
        x, y = testdata
        x = x.cuda()
        y = y.cuda()

        abu, output = net(x, y)

    # compute metric
    abu_est = torch.reshape(abu.squeeze(-1).permute(2,1,0), (num_classes,row,col)).permute(0,2,1).cpu().data.numpy()
    edm_result = torch.reshape(net.decoder[0].weight, (band,num_classes)).cpu().data.numpy()  
    abu_est = abu_est[position,:,:]
    edm_result = edm_result[:,position]

    RMSE = compute_rmse(label, abu_est)
    SAD = compute_sad(M_true, edm_result)
    print('**********************************')
    print('RMSE: {:.5f} | SAD: {:.5f}'.format(RMSE, SAD))
    print('**********************************')
    print('total computational cost:', time_end-time_start)
    print('**********************************')

    # abundance map
    for i in range(abu_est.shape[0]):
        plt.subplot(2, num_classes, i+1)
        plt.imshow(abu_est[i,:,:])
        plt.subplot(2, num_classes, i+1+num_classes)
        plt.imshow(label[i,:,:])
    plt.show()

    # print hyperparameter setting and save result 
    print_args(vars(args))
    save_path = str(args.dataset) + '_result.mat'
    sio.savemat(save_path, {'abu_est':abu_est.T, 'M_est':edm_result})
