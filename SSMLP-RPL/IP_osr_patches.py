import os
import argparse
import datetime
import time
import importlib
import scipy.io as sio
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from models import network

from core import train, test
from core.full_test import full_test

from generate_pic import aa_and_each_accuracy, sampling1, sampling2, sampling3, load_dataset, generate_png, generate_iter
from generate_pic import generate_train_iter, generate_valida_iter, generate_test_iter, generate_all_iter, generate_full_iter, generate_iter,generate_train_known_iter,generate_test_known_iter,generate_test_unknown_iter,generate_fulltest_iter
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', choices=['SA', 'PU', 'IP'], default='IP', help='dataset to use')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--num', type=int, default=30, help='number of samples')


# optimization
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--experiment_num', type=int, default=10)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='SSMLP-RPL')
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--embed_dims', type=int, default=64)
parser.add_argument('--segment_dim', type=int, default=8)


# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)


def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros(int(y_true.max() + 1))
    num = np.zeros(int(y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[int(y_true[i])] += 1
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num[int(y_pred[i])] += 1
    for i in range(len(num)):
        num[i] = num[i] / num_perclass[i]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ac = np.zeros(int(y_true.max() + 1 + 2))
    ac[:int(y_true.max() + 1)] = num
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()

    #path = 'D:/HSI_data/'
    #path = '/media/lenovo/0A08DBEC08DBD533/syf/HSI_data/'
    path = '/home/lenovo/data/syf/HSI_data/'
    #path='/media/liubing/cc5992d7-d217-4ded-97b4-fd47f4fa55f4/syf/HSI_data/'

    dataname=args.dataset
    model_name='SSMLP-RPL'

    data_hsi=sio.loadmat(path+'Indian_pines_corrected.mat')
    data_hsi=data_hsi['indian_pines_corrected']
    gt=sio.loadmat(path+'Indian_pines_gt.mat')
    gt=gt['indian_pines_gt']

    # training samples per class
    SAMPLES_NUM = args.num
    experiment_num=args.experiment_num
    ROWS, COLUMNS, BAND = data_hsi.shape
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
    gt2 = gt.reshape(np.prod(gt.shape[:2]), )
    CLASSES_NUM = gt.max()
    print('The class numbers of the HSI data is:', CLASSES_NUM)

    known = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # define the known classes
    print('The known class of the HSI data is:', known)
    unknown = list(set(list(range(0,  gt.max()))) - set(known))
    print('The unknown class of the HSI data is:', unknown)

    print('-----Importing Setting Parameters-----')

    patch_list=[7]

    for i_patch in range(len(patch_list)):
        patches=patch_list[i_patch]
        PATCH_LENGTH = int((patches-1)/2)
        # number of training samples per class
        # lr, num_epochs, batch_size = 0.0001, 200, 32



        img_rows = 2 * PATCH_LENGTH + 1
        img_cols = 2 * PATCH_LENGTH + 1
        INPUT_DIMENSION = data_hsi.shape[2]
        FULL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
        ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]



        data = preprocessing.scale(data)
        whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])


        padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),'symmetric')

        Experiment_result = np.zeros([CLASSES_NUM + 7, experiment_num + 2])
        for iter_num in range(experiment_num):


            np.random.seed(iter_num+123456)
            train_indices, test_indices = sampling1(SAMPLES_NUM, gt2, options['dataset'])
            full_indices = sampling3(gt2, 1)
            TRAIN_SIZE = len(train_indices)
            VAL_SIZE = int(TRAIN_SIZE)
            TEST_SIZE = len(test_indices)
            if  unknown == None:
                #full_iter = generate_full_iter(whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, args.batch_size, gt2, FULL_SIZE,full_indices)
                train_iter = generate_train_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,args.batch_size, gt2)
                test_iter = generate_test_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2)
                known_test_iter = None
                unknown_test_iter = None
            else:
                full_iter = generate_full_iter(whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, args.batch_size, gt2,FULL_SIZE, full_indices)
                test_iter = generate_fulltest_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known, CLASSES_NUM-1)
                train_iter = generate_train_known_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known, augmentation=True)
                known_test_iter = generate_test_known_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known)
                unknown_test_iter = generate_test_unknown_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, unknown, CLASSES_NUM-1)

            options.update(
                {
                    'BAND':BAND,
                    'known': known,
                    'unknown': unknown,
                    'full_iter' : full_iter,
                    'test_iter' : test_iter,
                    'train_iter': train_iter,
                    'known_test_iter' : known_test_iter,
                    'unknown_test_iter' : unknown_test_iter,

            }
            )


            torch.manual_seed(options['seed'])
            os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
            use_gpu = torch.cuda.is_available()
            if options['use_cpu']: use_gpu = False

            if use_gpu:
                print("Currently using GPU: {}".format(options['gpu']))
                cudnn.benchmark = True
                torch.cuda.manual_seed_all(options['seed'])
            else:
                print("Currently using CPU")

            # Dataset
            print("{} Preparation".format(options['dataset']))
            if options['unknown'] == None:
                trainloader, testloader, outloader = options['train_iter'], options['test_iter'], None
            else:
                trainloader, full_testloader, full_loader, testloader, outloader = options['train_iter'], options['test_iter'], \
                                                                                   options['full_iter'], options[
                                                                                       'known_test_iter'], options[
                                                                                       'unknown_test_iter']

            # unknow=options['unknown'][0]
            # Model
            print("Creating model: {}".format(options['model']))

            options['num_classes'] = len(options['known'])
            unknow = options['num_classes']

            net = network.SSMLP(patches, options['BAND'], options['num_classes'], layers=options['layers'], embed_dims=options['embed_dims'],segment_dim=options['segment_dim'])
            feat_dim = options['embed_dims']
            # feat_dim = 128

            # Loss
            options.update(
                {
                    'feat_dim': feat_dim,
                    'use_gpu': use_gpu
                }
            )

            Loss = importlib.import_module('loss.' + options['loss'])
            criterion = getattr(Loss, options['loss'])(**options)

            if use_gpu:
                net = nn.DataParallel(net).cuda()
                criterion = criterion.cuda()



            params_list = [{'params': net.parameters()},
                           {'params': criterion.parameters()}]

            # optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.Adam(params_list, lr=options['lr'])

            if options['stepsize'] > 0:
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

            start_time = time.time()

            for epoch in range(options['max_epoch']):
                print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

                _, logits_min, dis_min, loss_r = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

                #if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
                #    print("==> Test", options['loss'])
                    #results, pred, label = test(net, criterion, full_testloader, testloader, outloader, logits_min, dis_min,
                    #                            loss_r, unknow, epoch=epoch, **options)
                    #print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                    #                                                                        results['OSCR']))

                    # save_networks(net, model_path, file_name, criterion=criterion)
                if options['stepsize'] > 0: scheduler.step()

            train_time2=time.time()
            tes_time1=time.time()
            results, pred, label = test(net, criterion, full_testloader, testloader, outloader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],results['OSCR']))
            tes_time2=time.time()

            #pred_global = full_test(net, criterion, full_testloader, full_loader, testloader, outloader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)
            #generate_png(gt, pred_global, dataname, ROWS, COLUMNS, SAMPLES_NUM)

            ac = get_accuracy(label, pred)
            # ac2 = get_accuracy(_labels, _pred)

            elapsed = round(time.time() - start_time)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

            Experiment_result[0, iter_num] = ac[-1] * 100  # OA
            Experiment_result[1, iter_num] = np.mean(ac[:-2]) * 100  # AA
            Experiment_result[2, iter_num] = ac[-2] * 100  # Kappa
            Experiment_result[3, iter_num] = results['ACC']                           #Closed-set  OA
            Experiment_result[4, iter_num] = results['OSCR']                          #OSCR
            Experiment_result[5, iter_num] = train_time2 - start_time
            Experiment_result[6, iter_num] = tes_time2 - tes_time1
            Experiment_result[7:, iter_num] = ac[:-2] * 100

            print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))

            ########## mean value & standard deviation #############

        Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
        Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差


        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')

        f = open('./record/' + dataname + '/' + str(day_str) + '_' + dataname + '_' + model_name + '_' +str(SAMPLES_NUM)+ '_num_patch='+str(patches)+'_layers='+str(options['layers'])+'_embed_dims='+str(options['embed_dims'])+'_segment_dim='+str(options['segment_dim'])+'.txt','w')
        for i in range(Experiment_result.shape[0]):
            f.write(str(i + 1) + ':' + str(round(Experiment_result[i, -2], 2)) + '+/-' + str(
                round(Experiment_result[i, -1], 2)) + '\n')
        for i in range(Experiment_result.shape[1] - 2):
            f.write('Experiment_num' + str(i) + '_open_OA:' + str(Experiment_result[0, i]) + '\n')
        for i in range(Experiment_result.shape[1] - 2):
            f.write('Experiment_num' + str(i) + '_closed_OA:' + str(Experiment_result[3, i]) + '\n')
        f.close()
