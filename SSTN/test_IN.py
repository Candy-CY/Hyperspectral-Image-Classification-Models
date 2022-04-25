import os
import scipy.io as sio

import sys
sys.path.append('/home/zilong/SSTN')    # add the SSTN root path to environment path

import torch
import utils
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import time
import collections
import logging
import argparse

import torch
from torch.utils import data

from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from utils import cal_results, predVisIN
import collections

from NetworksBlocks import SSNet_AEAE_IN, SSRN

parser = argparse.ArgumentParser("IN")
# parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
# parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.002, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--sample', type=int, default=200, help='sample sizes for training')
parser.add_argument('--model', type=str, default='SSTN', help='select network to train')
parser.add_argument('--phi', type=str, default='AEAE', help='sequential order of network')
#parser.add_argument('--inorder', type=int, default=1, help='sequential order of input data')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

np.random.seed(2)
cudnn.benchmark = True
torch.manual_seed(2)

cudnn.enabled=True
torch.cuda.manual_seed(2)


args.save = 'IN-train-model-{}-arch-{}-{}-lr{}'.format(args.model, args.phi, time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('train_IN.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return whole_indices, train_indices, test_indices



sample_200 = [2, 27, 19, 4, 9, 14, 2, 10, 3, 24, 41, 14, 4, 18, 7, 2]
rsample_200 = [1, 28, 16, 5, 9, 14, 1, 9, 1, 19, 47, 12, 4, 24, 8, 2]

def rsampling(groundTruth, sample_num = sample_200, rsample_num = rsample_200):              #divide dataset into train and test datasets
    labels_loc = {}
    labeled = {}
    train2 = {}
    val = {}
    test = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        labeled[i] = indices[:sample_num[i]]
        train2[i] = indices[sample_num[i]:sample_num[i]+rsample_num[i]]
        val[i] =  indices[-(sample_num[i]+rsample_num[i]):]
        test[i] = indices[sample_num[i]+rsample_num[i]:-(sample_num[i]+rsample_num[i])]
    whole_indices = []
    labeled_indices = []
    train2_indices = []
    val_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        labeled_indices += labeled[i]
        train2_indices += train2[i]
        val_indices += val[i]
        test_indices += test[i]
        np.random.shuffle(labeled_indices)
        np.random.shuffle(train2_indices)        
        np.random.shuffle(val_indices)        
        np.random.shuffle(test_indices)
    return whole_indices, labeled_indices, train2_indices, val_indices, test_indices


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix


IN_PATH = './datasets'
mat_data = sio.loadmat(IN_PATH + '/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat(IN_PATH + '/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
#print (data_IN.shape)

# Input dataset configuration to generate 103x7x7 HSI samples
new_gt_IN = gt_IN

#batch_size = 16
nb_classes = 9
#img_rows, img_cols =  7, 7 # 9, 9        

INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION = 200

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 10249
# VAL_SIZE = 4281

TRAIN_SIZE = 200 #300
DEV_SIZE = 200
VAL_SIZE = 400
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - DEV_SIZE - VAL_SIZE
# VALIDATION_SPLIT = 0.9                      # 20% for trainnig and 80% for validation and testing

img_channels = 200
PATCH_LENGTH = 4                #Patch_size 9*9

MAX = data_IN.max()
data_IN = np.transpose(data_IN, (2,0,1))

data_IN = data_IN - np.mean(data_IN, axis=(1,2), keepdims=True)
data_IN = data_IN / MAX

data = data_IN.reshape(np.prod(data_IN.shape[:1]),np.prod(data_IN.shape[1:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)

whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
#whole_data = whole_data - np.mean(whole_data, axis=(1,2), keepdims=True)
padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

#CATEGORY = 9

train_data = np.zeros((TRAIN_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
test_data = np.zeros((TEST_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
all_data = np.zeros((TOTAL_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))


all_indices, train_indices, dev_indices, val_indices, test_indices = rsampling(gt)

y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1
y_all = gt[all_indices] - 1

train_assign = indexToAssignment(train_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(train_assign)):
    train_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[i][0], train_assign[i][1])
    
test_assign = indexToAssignment(test_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(test_assign)):
    test_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[i][0], test_assign[i][1])
        
all_assign = indexToAssignment(all_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(all_assign)):
    all_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i][0], all_assign[i][1])

import torch
from torch.utils import data

class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels):
        
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]

        return X, y


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device('cuda', args.gpu)
#torch.cudnn.benchmark = True

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8}
max_epochs = 100

# Generators
training_set = HSIDataset(range(len(train_indices)), train_data, y_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = HSIDataset(range(len(test_indices)), test_data, y_test)
validation_generator = data.DataLoader(validation_set, **params)

all_set = HSIDataset(range(len(all_indices)), all_data, y_all)
all_generator = data.DataLoader(all_set, **params)


trainloader = torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True, num_workers=8)

validationloader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=False, num_workers=8)

allloader = torch.utils.data.DataLoader(all_set, batch_size=50, shuffle=False, num_workers=8)

SAVE_PATH3 = './pretrained_models/IN-' + str(args.model) + '/' + str(args.model) + '_sample200.pth' # args.save + '/' + str(args.model) + '_sample' + str(args.sample) + '.pth' 

# Validation Stage
if args.model == 'SSTN':
    trained_net = SSNet_AEAE_IN()
elif args.model == 'SSRN':
    trained_net = SSRN(num_classes=16, k=97)
else:
    logging.error("No such model in our zoo!")


trained_net.load_state_dict(torch.load(SAVE_PATH3))
trained_net.eval()
trained_net = trained_net.cuda()

label_val = []
pred_val = []

with torch.no_grad():
    for data in validationloader:
        images, labels = data
        #label_val = torch.stack([label_val.type_as(labels), labels])
        label_val.append(labels)
        
        images, labels = images.to(device), labels.to(device)
        outputs = trained_net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        #pred_val = torch.stack([pred_val.type_as(predicted), predicted])
        pred_val.append(predicted)
        
label_val_cpu = [x.cpu() for x in label_val]
pred_val_cpu = [x.cpu() for x in pred_val]

label_cat = np.concatenate(label_val_cpu)
pred_cat = np.concatenate(pred_val_cpu)

matrix = metrics.confusion_matrix(label_cat, pred_cat)

OA, AA_mean, Kappa, AA = cal_results(matrix)

logging.info('OA, AA_Mean, Kappa: %f, %f, %f, ', OA, AA_mean, Kappa)
logging.info(str(("AA for each class: ", AA)))


# generate classification maps

all_pred = []

with torch.no_grad():
     for data in allloader:
         images, _ = data
         images, _ = images.to(device), labels.to(device)
         outputs = trained_net(images.float())
         _, predicted = torch.max(outputs.data, 1)
         all_pred.append(predicted)

all_pred = torch.cat(all_pred)
all_pred = all_pred.cpu().numpy() + 1

y_pred = predVisIN(all_indices, all_pred, 145, 145)


#plt.plot(x, y)
plt.imshow(y_pred)
plt.axis('off')
fig_path = './Cmaps/IN-' + str(args.model) + '.png'
plt.savefig(fig_path, bbox_inches=0)
#plt.savefig(fig_path, bbox_inches='tight')



