import os
import argparse
import time
import numpy as np
#import matplotlib
# import matplotlib as mpl # use slurm
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
import cv2
import apex
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import cohen_kappa_score
#from thop import profile
from func import load,product,intersectionAndUnionGPU

## GPU_configration

# USE_GPU=True
# if USE_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# else:
#     device=torch.device('cpu')
#     print('using device:',device)

torch.backends.cudnn.benchmark = False

def main():
    ############ parameters setting ############

    parser = argparse.ArgumentParser(description="Network Trn_val_Tes")
    ## dataset setting
    parser.add_argument('--dataset', type=str, default='indian',
                        choices=['indian','pavia','houston','salina','ksc'],
                        help='dataset name')
    ## network setting
    parser.add_argument('--network', type=str, default='sgrhsi',
                        choices=['segrn','sagrn','ssgrn','fcn'],
                        help='network name')
    ## normalization setting
    parser.add_argument('--norm', type=str, default='std',
                        choices=['std','norm'],
                        help='nomalization mode')
    parser.add_argument('--mi', type=int, default=-1,
                        help='min normalization range')
    parser.add_argument('--ma', type=int, default=1,
                        help='max normalization range')
    ## experimental setting
    parser.add_argument('--sync_bn', type=str, default='True',
                        choices=['True', 'False'],help='synchronized batchNorm')
    parser.add_argument('--use_apex', type=str, default='False',
                        choices=['True', 'False'],help='mixed-precision training')
    parser.add_argument('--opt_level', type=str, default='O1',
                        choices=['O0', 'O1','O2'], help='mixed-precision')
    parser.add_argument('--input_mode', type=str, default='part',
                        choices=['whole', 'part'],help='input setting')
    parser.add_argument('--input_size', nargs='+', type=int)
    parser.add_argument('--overlap_size', type=int, default=16,
                        help='size of overlap')
    parser.add_argument('--experiment-num', type=int, default=1,
                        help='experiment trials number')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=4,
                        help='input batch size for validation')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--workers', type=int, default=2,
                        help='workers num')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='ignore label')
    parser.add_argument('--print_freq', type=int, default=3,
                        help='print frequency')
    parser.add_argument("--resume", type=str, help="model path.")
    # model setting
    parser.add_argument('--sa_groups', type=int, default=256, help='spatial group number')
    parser.add_argument('--se_groups', type=int, default=256, help='spectral group number')

    args = parser.parse_args()

    ############# load dataset(indian_pines & pavia_univ...)######################

    a=load()

    All_data,labeled_data,rows_num,categories,r,c,FLAG=a.load_data(flag=args.dataset)

    print('Data has been loaded successfully!')

    ##################### normlization ######################

    if args.norm=='norm':
        scaler = MinMaxScaler(feature_range=(args.mi,args.ma))
        All_data_norm=scaler.fit_transform(All_data[:,1:-1])

    elif args.norm=='std':
        scaler = StandardScaler()
        All_data_norm = scaler.fit_transform(All_data[:, 1:-1])

    print('Image normlization successfully!')

    ########################### Data preparation ##################################


    if args.input_mode=='whole':

        X_data=All_data_norm.reshape(1,r,c,-1)

        args.print_freq=1

        args.input_size=[r,c]

    elif args.input_mode=='part':


        image_size=(r, c)

        input_size=args.input_size

        LyEnd,LxEnd = np.subtract(image_size, input_size)

        Lx = np.linspace(0, LxEnd, int(np.ceil(LxEnd/np.float(input_size[1]-args.overlap_size)))+1, endpoint=True).astype('int')
        Ly = np.linspace(0, LyEnd, int(np.ceil(LyEnd/np.float(input_size[0]-args.overlap_size)))+1, endpoint=True).astype('int')

        image_3D=All_data_norm.reshape(r,c,-1)

        N=len(Ly)*len(Lx)

        X_data = np.zeros([N,input_size[0],input_size[1],image_3D.shape[-1]])#N,H,W,C

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart,cStart = (Ly[j],Lx[k])
                rEnd,cEnd = (rStart+input_size[0],cStart+input_size[1])
                X_data[i] = image_3D[rStart:rEnd,cStart:cEnd,:]
                i+=1
    else:
        raise NotImplementedError

    print('{} image preparation Finished!, Data Number {}, '
          'Data size ({},{})'.format(args.dataset,X_data.shape[0],X_data.shape[1],X_data.shape[2]))

    X_data = torch.from_numpy(X_data.transpose(0, 3, 1, 2))#N,C,H,W

    ##################################### trn/val/tes ####################################

    #Experimental memory
    Experiment_result=np.zeros([categories+4,12])#OA,AA,kappa,trn_time,tes_time

    #kappa
    kappa=0

    y_map=All_data[:, -1].reshape(r,c)

    if args.network == 'ssgrn':
        print('Implementing Spectral-Spatial Graph Attention Network!')
    elif args.network == 'segrn':
        print('Implementing Spectral Graph Attention Network!')
    elif args.network == 'sagrn':
        print('Implementing Spatial Graph Attention Network!')
    elif args.network == 'fcn':
        print('Implementing Fully Convolutional Network!')
    else:
        raise NotImplementedError
        
    for count in range(0, args.experiment_num):

        a = product(c, FLAG, All_data)

        rows_num,trn_num,val_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num)

        #################################### trn_label #####################################

        y_trn_map=a.production_label(trn_num, y_map, split='Trn')

        # plt.xlabel('trn_label_map')
        # plt.imshow(y_trn_map,cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        if args.input_mode == 'whole':

            y_trn_data=y_trn_map.reshape(1,r,c)

        elif args.input_mode=='part':

            y_trn_data = np.zeros([N, input_size[0], input_size[1]], dtype=np.int32)  # N,H,W

            i=0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = Ly[j], Lx[k]
                    rEnd, cEnd = rStart + input_size[0], cStart + input_size[1]
                    y_trn_data[i] = y_trn_map[rStart:rEnd, cStart:cEnd]
                    i+=1
        else:
            raise NotImplementedError


        # plt.xlabel('trn_data_map')
        # plt.imshow(y_trn_data[0], cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_trn_data-=1

        y_trn_data[y_trn_data<0]=255

        y_trn_data = torch.from_numpy(y_trn_data)

        print('Experiment {}，training dataset preparation Finished!'.format(count))

        #################################### val_label #####################################

        y_val_map = a.production_label(val_num, y_map, split='Val')

        # plt.xlabel('val_label_map')
        # plt.imshow(y_val_map, cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        if args.input_mode == 'whole':

            y_val_data = y_val_map.reshape(1, r, c)

        elif args.input_mode == 'part':

            y_val_data = np.zeros([N, input_size[0], input_size[1]])  # N,H,W

            i=0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    y_val_data[i,:,:] = y_val_map[rStart:rEnd, cStart:cEnd]
                    i+=1
        else:
            raise NotImplementedError

        # plt.xlabel('val_data_map')
        # plt.imshow(y_val_data[0], cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_val_data -= 1

        y_val_data[y_val_data < 0] = 255

        y_val_data = torch.from_numpy(y_val_data)

        print('Experiment {}，validation dataset preparation Finished!'.format(count))

        ########## training/Validation #############

        torch.cuda.empty_cache()#GPU memory released

        trn_dataset=TensorDataset(X_data, y_trn_data)

        trn_loader=DataLoader(trn_dataset,batch_size=args.batch_size,num_workers=args.workers,
                              shuffle=True, drop_last=True, pin_memory=False)

        val_dataset = TensorDataset(X_data, y_val_data)

        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,shuffle=False, pin_memory=False)

        if args.network=='segrn' or args.network=='sagrn' or args.network=='ssgrn' or args.network=='fcn':
            from SSGRN import SSGRN
            net = SSGRN(args, X_data.shape[1],categories-1)
        else:
            raise NotImplementedError

        params = [dict(params=net.parameters(), lr=args.lr)]

        optimizer = torch.optim.SGD(params, momentum=0.9,
                                    lr=args.lr, weight_decay=args.weight_decay)

        if args.use_apex=='True':# use apex

            net, optimizer = apex.amp.initialize(net.cuda(), optimizer, opt_level=args.opt_level)

        net= torch.nn.DataParallel(net.cuda())

        #net.cuda()

        #patch_replication_callback(net)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        trn_time=0
        best_val_OA=0

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                print("=> loading ft model...")
                ckpt_dict = checkpoint.state_dict()
                model_dict = {}
                state_dict = net.state_dict()
                for k, v in ckpt_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                net.load_state_dict(state_dict)

                print("=> loaded checkpoint '{}' ".format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                raise NotImplementedError

        for i in range(0, args.epochs):
            trn_time1 = time.time()
            train(args, i, net, optimizer, trn_loader, criterion, categories)
            trn_time2 = time.time()
            trn_time = trn_time + trn_time2 - trn_time1
            if (i+1) % 100==0:
                val_OA = validation(args, i, net, val_loader, categories)

                if val_OA >= best_val_OA:
                    filename = str(args.network) + '_' + str(FLAG) + '_' + 'experiment_{}'.format(count) + '_valbest_tmp' + '.pth'
                    torch.save(net, filename)

        print('########### Experiment {}，Model Training Period Finished! ############'.format(count))

        #################################### test_label ####################################

        y_tes_map = a.production_label(tes_num, y_map, split='Tes')

        # plt.xlabel('tes_label_map')
        #
        # plt.imshow(y_tes_map, cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_tes_data = y_tes_map.reshape(r, c)

        y_tes_data -= 1

        y_tes_data[y_tes_data < 0] = 255

        print('Experiment {}，Testing dataset preparation Finished!'.format(count))

        ################### testing ################

        filename = str(args.network) + '_' + str(FLAG) + '_' + 'experiment_{}'.format(count) + '_valbest_tmp' + '.pth'
        net = torch.load(filename, map_location='cpu')
        net = net.cuda()

        tes_time1 = time.time()

        if args.input_mode == 'whole':

            net.eval()
            with torch.no_grad():
                pred = net(X_data.float())
                pred = pred[0].cpu().numpy()
                y_tes_pred = np.argmax(pred, 1).squeeze(0)

        elif args.input_mode == 'part':

            img=torch.from_numpy(image_3D).permute(2,0,1) #C,H,W
            y_tes_pred = np.zeros([r, c])
            net.eval()

            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    img_part = img[:,rStart:rEnd,cStart:cEnd].unsqueeze(0)
                    with torch.no_grad():
                        pred = net(img_part.float())
                    pred = pred[0].cpu().numpy()
                    pred = np.argmax(pred,1).squeeze(0)

                    if j == 0 and k == 0:
                        y_tes_pred[rStart:rEnd, cStart:cEnd] = pred
                    elif j == 0 and k > 0:
                        y_tes_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = pred[:,
                                                                                           int(args.overlap_size / 2):]
                    elif j > 0 and k == 0:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = pred[
                                                                                           int(args.overlap_size / 2):,
                                                                                           :]
                    else:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd,
                        cStart + int(args.overlap_size / 2):cEnd] = pred[int(args.overlap_size / 2):,
                                                                    int(args.overlap_size / 2):]
        else:
            raise NotImplementedError

        tes_time2 = time.time()

        print('########### Experiment {}，Model Testing Period Finished! ############'.format(count))

        ####################################### assess ###########################################

        y_tes_data_1d = y_tes_data.reshape(r*c)
        y_tes_pred_1d = y_tes_pred.reshape(r*c)

        y_tes_gt=y_tes_data_1d[tes_num]
        y_tes=y_tes_pred_1d[tes_num]

        print('==================Test set=====================')
        print('Experiment {}，Testing set OA={}'.format(count,np.mean(y_tes_gt==y_tes)))
        print('Experiment {}，Testing set Kappa={}'.format(count,cohen_kappa_score(y_tes_gt,y_tes)))

        if cohen_kappa_score(y_tes_gt,y_tes)>=kappa or count==0:
            if args.resume:
                torch.save(net, str(args.network) + '_' + str(FLAG) + '_groups_'+ str(args.sa_groups)+'.pth')
                kappa = cohen_kappa_score(y_tes_gt, y_tes)
                np.save(str(args.network) + '_' + str(FLAG) + '_groups_'+ str(args.sa_groups)+'.npy', y_tes_pred)
            else:
                torch.save(net,str(args.network)+'_'+str(FLAG) + '_groups_'+ str(args.sa_groups)+ '.pth')
                kappa=cohen_kappa_score(y_tes_gt,y_tes)
                np.save(str(args.network) + '_' + str(FLAG) +'_groups_'+ str(args.sa_groups)+'.npy',y_tes_pred)

        ## Detailed information (every class accuracy)

        num_tes=np.zeros([categories-1])
        num_tes_pred=np.zeros([categories-1])
        for k in y_tes_gt:
            num_tes[int(k)]+=1# class index start from 0
        for j in range(y_tes_gt.shape[0]):
            if y_tes_gt[j]==y_tes[j]:
                num_tes_pred[int(y_tes_gt[j])]+=1

        Acc=num_tes_pred/num_tes*100

        Experiment_result[0,count]=np.mean(y_tes_gt==y_tes)*100#OA
        Experiment_result[1,count]=np.mean(Acc)#AA
        Experiment_result[2,count]=cohen_kappa_score(y_tes_gt,y_tes)*100#Kappa
        Experiment_result[3, count] = trn_time
        Experiment_result[4, count] = tes_time2 - tes_time1
        Experiment_result[5:,count]=Acc

        print('Experiment {}，Testing set AA={}'.format(count, np.mean(Acc)))

        for i in range(categories - 1):
            print('Class_{}: accuracy {:.4f}.'.format(i + 1, Acc[i]))

        print('########### Experiment {}，Model assessment Finished！ ###########'.format(count))

    ########## mean value & standard deviation #############

    Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
    Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

    if args.resume:
        scio.savemat(str(args.network) + '_' + str(FLAG) +'_groups_'+ str(args.sa_groups)+ '.mat', {'data': Experiment_result})

        y_disp_all = np.load(str(args.network) + '_' + str(FLAG) +'_groups_'+ str(args.sa_groups)+ '.npy')

        cv2.imwrite(str(args.network) + '_' + str(FLAG) +'_groups_'+ str(args.sa_groups)+ '.png', y_disp_all.reshape(r, c))
    else:
        scio.savemat(str(args.network)+'_sample_'+str(FLAG) +'_groups_'+ str(args.sa_groups)+'_'+str(args.experiment_num)+'.mat',{'data':Experiment_result})

        y_disp_all = np.load(str(args.network) + '_' + str(FLAG) +'_groups_'+ str(args.sa_groups)+ '.npy')

        cv2.imwrite(str(args.network)+'_'+str(FLAG) +'_groups_'+ str(args.sa_groups)+ '.png', y_disp_all.reshape(r,c))

    # plt.xlabel('pre image')
    # plt.imshow(y_disp_all.reshape(r, c), cmap='jet')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.show()

    print('One time training cost {:.4f} secs'.format(trn_time))
    print('One time testing cost {:.4f} secs'.format(tes_time2 - tes_time1))

    print('Results Saving Finished!')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fix_gn(m):
    classname = m.__class__.__name__
    if classname.find('GroupNorm') != -1:
        m.eval()

def train(args, epoch, net, optimizer, trn_loader, criterion, categories):
    net.train()  # train mode

    if args.resume:
        net.apply(fix_gn)

    max_iter=args.epochs * len(trn_loader)
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    target_meter = AverageMeter()
    for idx, (X_data, y_target) in enumerate(trn_loader):

        X_data=X_data.float().cuda()
        y_target = y_target.float().cuda()

        y_pred = net.forward(X_data)

        y_target= y_target.long()

        for i in range(len(y_pred)):
            if i == 0:
                loss = criterion(y_pred[i], y_target)
            if i > 0:
                # if i==len(y_pred)-1:
                #     loss += 0.4*criterion(y_pred[i], y_target)
                # else:
                loss += criterion(y_pred[i], y_target)

        _, predicted = torch.max(y_pred[0], 1)

        # back propagation
        optimizer.zero_grad()
        if args.use_apex=='True':
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # compute acc
        n = X_data.size(0)  # batch size
        loss_meter.update(loss.item(), n)
        intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories-1, args.ignore_label)
        intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # oa of a bs

        del X_data, y_target
        del y_pred

        # updata lr
        if args.resume:
            current_lr = args.lr
        else:
            current_iter = epoch * len(trn_loader) + idx + 1
            current_lr = args.lr * (1 - float(current_iter) / max_iter) ** 0.9

        optimizer.param_groups[0]['lr'] = current_lr

        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}], '
                  'Batch Loss {loss_meter.val:.4f}, '
                  'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, idx + 1, len(trn_loader),
                                                    loss_meter=loss_meter,accuracy=accuracy))

    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Training epoch [{}/{}]: Loss {:.4f} AA/OA {:.4f}/{:.4f}.'.format(epoch + 1,
                                                                args.epochs,loss_meter.avg,
                                                                mAcc, allAcc))

def validation(args, epoch, net, val_loader, categories):
    print('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    net.eval()  # evaluation mode
    intersection_meter = AverageMeter()
    target_meter = AverageMeter()

    for idx, (X_data, y_target) in enumerate(val_loader):
        with torch.no_grad():
            X_data = X_data.float().cuda()
            y_target = y_target.float().cuda()
            y_pred = net.forward(X_data)

        _, predicted = torch.max(y_pred[0], 1)

        y_target = y_target.long()

        # compute acc
        intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories - 1, args.ignore_label)
        intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # oa of a bs

        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}], '
                  'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, idx + 1,
                                                    len(val_loader),accuracy=accuracy))
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Validation epoch [{}/{}]: AA/OA {:.4f}/{:.4f}.'.format(epoch + 1,
                                                                args.epochs, mAcc, allAcc))
    for i in range(categories-1):
        print('Class_{}: accuracy {:.4f}.'.format(i+1, accuracy_class[i]))

    print('>>>>>>>>>>>>>>>> End Evaluation <<<<<<<<<<<<<<<<<<')

    return allAcc

if __name__ == '__main__':
    main()
