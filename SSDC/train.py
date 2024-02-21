import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import torch.optim as optim
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from collections import Counter
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models
import da_net_doconv as da_net
import spectral
import heapq
from conv import DOConv2d, DOConv2d_multi
from torch.utils.data.sampler import WeightedRandomSampler
import tst_net



parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200) 
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 9)
parser.add_argument("-b","--query_num_per_class",type = int, default = 18)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-trb","--train_batch_size",type=int, default=432)
parser.add_argument("-teb","--test_batch_size",type=int, default=432)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-p","--patch_size",type=int,default=9)
parser.add_argument('-momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('-l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('-lr', type=float, default=1e-2,
                    help="Learning rate, set by the model if not specified.")
# target
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=18, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
PATCH_SIZE = args.patch_size
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
TRAIN_BATCH_SIZE = args.train_batch_size
TEST_BATCH_SIZE = args.test_batch_size

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('./dataset/source/',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)

# process source domain data set
data_train = source_imdb['data']
labels_train = source_imdb['Labels']

keys_all_train = sorted(list(set(labels_train)))

label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) 
print("Num classes of the number of class larger than 0: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
del data

# source domain adaptation data
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print("source_imdb['data']",source_imdb['data'].shape)
print("source_imdb['labels']",source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)

bal_source_dataset = source_imdb['Labels']
source_count_dict = Counter(bal_source_dataset)

source_count_dict_full = { lbl:0 for lbl in range(CLASS_NUM)}
for k, v in source_count_dict.items(): source_count_dict_full[k] = v
source_count_dict_sorted = {k: v for k, v in sorted(source_count_dict_full.items(), key=lambda item: item[0])}
source_class_sample_count = np.array(list(source_count_dict_sorted.values()))
source_class_sample_count = source_class_sample_count / source_class_sample_count.max()
source_class_sample_count += 1e-8
source_weights = 1 / torch.Tensor(source_class_sample_count)
source_sample_weights = [source_weights[l] for l in bal_source_dataset]
source_sample_weights = torch.DoubleTensor(np.array(source_sample_weights))
source_sampler = WeightedRandomSampler(source_sample_weights, len(source_sample_weights)) 

source_loader = torch.utils.data.DataLoader(source_dataset, sampler=source_sampler, batch_size=TRAIN_BATCH_SIZE, num_workers=0)
del source_dataset, source_imdb

# target domain data set
# load target domain data set
test_data = './dataset/Indian_pines/indian_pines_corrected.mat'
test_label = './dataset/Indian_pines/indian_pines_gt.mat'
Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((1000 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)
    
    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm) 

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
        

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    
    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  
    train_datas, train_labels = train_loader.__iter__().next()

    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print('target data augmentation data:',target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set

    # target domain : batch samples for domian adaptation
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()
criterion_freq = nn.BCELoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    
    # model
    feature_encoder = da_net.Network()
    domain_classifier = models.DomainClassifier()
    graph_classifier = tst_net.Feature_Extractor(160, 128, CLASS_NUM)
    random_layer = models.RandomLayer([128, args.class_num], 1024)  

    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)
    graph_classifier.apply(weights_init)

    feature_encoder.cuda()
    domain_classifier.cuda()
    random_layer.cuda()
    graph_classifier.cuda()
  
    feature_encoder.train()
    domain_classifier.train()
    graph_classifier.train()
   
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (iDataSet) / nDataSet), 0.75)
    graph_classifier_optim = optim.SGD(graph_classifier.parameters(), lr=LEARNING_RATE, momentum=args.momentum, weight_decay = args.l2_decay)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    
    for episode in range(10000):
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()
            
        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  
            querys, query_labels = query_dataloader.__iter__().next() 
            
            # calculate features
            support_spa, support_features, support_outputs = feature_encoder(supports.cuda())  
            query_spa, query_features, query_outputs = feature_encoder(querys.cuda())    
            
            #target_data
            target_spa, target_features, target_outputs = feature_encoder(target_data.cuda(), domain='target')

            #graph prediction
            source_graph = torch.cat([support_spa, query_spa])   
            target_graph = target_spa   
            TST_wd, TST_gwd = graph_classifier(source_graph, target_graph)
           
            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  
            else:
                support_proto = support_features
                
            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())
            
            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)
            
            # set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label[:supports.shape[0] + querys.shape[0]] = 1  
          
            randomlayer_out = random_layer.forward([features, softmax_output])
            
            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)
            
            #svd
            features_source = torch.cat([support_features, query_features], dim=0)
            features_target = target_features
            svd_source = features_source.cpu().detach().numpy()
            svd_target = features_target.cpu().detach().numpy()
                        
            #source
            U_s, Sigma_s, Vt_s = np.linalg.svd(svd_source) 
            Sigma2_s = Sigma_s * Sigma_s
            loss_svds = np.sum(Sigma2_s)     
            
            #target
            U_t, Sigma_t, Vt_t = np.linalg.svd(svd_target)
            Sigma2_t = Sigma_t * Sigma_t
            loss_svdt = np.sum(Sigma2_t) 
            
            loss =  f_loss + domain_loss + TST_gwd + 0.65 * loss_svds + 0.35 * loss_svdt

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            graph_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()
            graph_classifier_optim.step()
            
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  
            querys, query_labels = query_dataloader.__iter__().next() 

            # calculate features
            support_spa, support_features, support_outputs = feature_encoder(supports.cuda(),  domain='target')  
            query_spa, query_features, query_outputs = feature_encoder(querys.cuda(), domain='target')  
            source_spa, source_features, source_outputs = feature_encoder(source_data.cuda()) 
                
            #graph prediction
            source_graph = source_spa
            target_graph = torch.cat([support_spa, query_spa])
            TST_wd, TST_gwd = graph_classifier(source_graph, target_graph)

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  
            else:
                support_proto = support_features
            
            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''domain adaptation'''
            features = torch.cat([support_features, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[supports.shape[0] + querys.shape[0]:] = 1 

            randomlayer_out = random_layer.forward([features, softmax_output]) 

            domain_logits = domain_classifier(randomlayer_out, episode)  
            domain_loss = domain_criterion(domain_logits, domain_label)

            #svd
            features_target = torch.cat([support_features, query_features], dim=0)
            features_source = source_features
            svd_source = features_source.cpu().detach().numpy()
            svd_target = features_target.cpu().detach().numpy()
            
            #source
            U_s, Sigma_s, Vt_s = np.linalg.svd(svd_source)
            Sigma2_s = Sigma_s * Sigma_s
            loss_svds = np.sum(Sigma2_s) 
            
            #target
            U_t, Sigma_t, Vt_t = np.linalg.svd(svd_target)
            Sigma2_t = Sigma_t * Sigma_t
            loss_svdt = np.sum(Sigma2_t)     
            
            loss =  f_loss + domain_loss + TST_gwd + 0.65 * loss_svds + 0.35 * loss_svdt 

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            graph_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()
            graph_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                domain_loss.item(),
                                                                                                                f_loss.item(),
                                                                                                                total_hit / total_num,
                                                                                                                loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            center_label = np.array([], dtype=np.int64)
            center_data = np.array([], dtype = np.int64)


            train_datas, train_labels = train_loader.__iter__().next()
            _, train_features, _= feature_encoder(Variable(train_datas).cuda(), domain='target') 

            max_value = train_features.max()  
            min_value = train_features.min() 
            print(max_value.item())
            print(min_value.item())
            train_features_reu = (train_features - min_value) * 1.0 / (max_value - min_value)
            
            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features_reu.cpu().detach().numpy(), train_labels)
                
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                _, test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target') 
                test_features_reu = (test_features - min_value) * 1.0 / (max_value - min_value)
                
                predict_labels = KNN_classifier.predict(test_features_reu.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str("checkpoints/6535" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
    #print('o',best_G.shape,best_RandPerm.shape,best_Row.shape, best_Column.shape,best_nTrain) #(618,348) (42776) (42776,) (42776,) 45
AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/IP.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
