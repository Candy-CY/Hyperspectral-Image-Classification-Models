import os
import torch
import torch.utils.data as dataf
import scipy.io as io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from model import CNN
import torch.nn as nn
from sklearn.manifold import TSNE
import auxil


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range




DATASETS_WITH_HSI_PARTS = ['Berlin', 'Augsburg']
DATA2_List = ['L','SAR','DSM','MS']
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
#
#datasetNames = ["Houston","Trento","MUUFL","Augsburg"]
datasetNames = ["Houston"]
testSizeNumber = 100
patchsize = 11
batchsize = 64
EPOCH = 500
LR = 5e-4
FM = 16
HSIOnly = True
FileName = "MorphTrans_TokenizationChannel_32tokens_DifferentKernelSizeMorphs"




# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


datasetName = "Houston"





print("----------------------------------Training for ",datasetName," ---------------------------------------------")
try:
    os.makedirs(datasetName)
except FileExistsError:
    pass

data1Name = datasetName

HSI = io.loadmat('./datasets/'+data1Name+'11x11/HSI_Tr.mat')
TrainPatch = HSI['Data']
TrainPatch = TrainPatch.astype(np.float32)
NC = TrainPatch.shape[3] # NC is number of bands

label = io.loadmat('./datasets/'+data1Name+'11x11/TrLabel.mat')
TrLabel = label['Data']

# Test data
if data1Name in DATASETS_WITH_HSI_PARTS:
    i = 2
    basePath = "./datasets/"+data1Name+'11x11/HSI_Te_Part'
    TestPatch = io.loadmat(basePath + str(i - 1) + '.mat')['Data']
    while True:
        my_file = Path(basePath + str(i) + '.mat')
        if my_file.exists():
            TestPatch = np.concatenate([TestPatch,io.loadmat(basePath + str(i) + '.mat')['Data']], axis = 0)
            i += 1
        else:
            break
else:
    HSI = io.loadmat('./datasets/'+data1Name+'11x11/HSI_Te.mat')
    TestPatch = HSI['Data']


TestPatch = TestPatch.astype(np.float32)

label = io.loadmat('./datasets/'+data1Name+'11x11/TeLabel.mat')
TsLabel = label['Data']


TrainPatch1 = torch.from_numpy(TrainPatch).to(torch.float32)
TrainPatch1 = TrainPatch1.permute(0,3,1,2)
TrainPatch1 = TrainPatch1.reshape(TrainPatch1.shape[0],TrainPatch1.shape[1],-1).to(torch.float32)
TrainLabel1 = torch.from_numpy(TrLabel)-1
TrainLabel1 = TrainLabel1.long()
TrainLabel1 = TrainLabel1.reshape(-1)

TestPatch1 = torch.from_numpy(TestPatch).to(torch.float32)
TestPatch1 = TestPatch1.permute(0,3,1,2)
TestPatch1 = TestPatch1.reshape(TestPatch1.shape[0],TestPatch1.shape[1],-1).to(torch.float32)
TestLabel1 = torch.from_numpy(TsLabel)-1
TestLabel1 = TestLabel1.long()
TestLabel1 = TestLabel1.reshape(-1)

Classes = len(np.unique(TrainLabel1))
dataset = dataf.TensorDataset(TrainPatch1,TrainLabel1)
if data1Name in ['Berlin']:
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers= 0)
else:
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers= 4)
print("HSI Train data shape = ", TrainPatch1.shape)
print("Train label shape = ", TrainLabel1.shape)

print("HSI Test data shape = ", TestPatch1.shape)
print("Test label shape = ", TestLabel1.shape)

print("Number of Classes = ", Classes)



KAPPA = []
OA = []
AA = []
ELEMENT_ACC = np.zeros((1, Classes))

set_seed(42)
for iterNum in range(1):
    cnn = CNN(FM, NC, Classes, HSIOnly).cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,weight_decay=5e-3)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    BestAcc = 0

    torch.cuda.synchronize()
    # train and test the designed model
    for epoch in range(EPOCH):
        for step, (b_x1, b_y) in enumerate(train_loader):
            # move train data to GPU
            b_x1 = b_x1.cuda()
            b_y = b_y.cuda()

            out1 = cnn(b_x1)
            loss = loss_func(out1, b_y)


            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                cnn.eval()
                pred_y = np.empty((len(TestLabel1)), dtype='float32')
                number = len(TestLabel1) // testSizeNumber
                for i in range(number):
                    temp = TestPatch1[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                    temp = temp.cuda()


                    temp2 = cnn(temp)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
                    del temp, temp2, temp3


                if (i + 1) * testSizeNumber < len(TestLabel1):
                    temp = TestPatch1[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                    temp = temp.cuda()
                    temp2 = cnn(temp)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cpu()
                    del temp, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % (accuracy*100))

                # save the parameters in network
                if accuracy > BestAcc:
                    BestAcc = accuracy
                    torch.save(cnn.state_dict(), datasetName+'/net_params_'+FileName+'_HSIOnly.pkl')
#                                     print("Weights = ", w0.cpu().detach().numpy())
#                                     print(w1.cpu().detach().numpy())


                cnn.train()
        scheduler.step()
    torch.cuda.synchronize()


    cnn.load_state_dict(torch.load(datasetName+'/net_params_'+FileName+'_HSIOnly.pkl'))


    cnn.eval()
    confusion, oa, each_acc, aa, kappa = auxil.reports(TestPatch1, TestLabel1, datasetName, cnn, testSizeNumber)
    print("OA AA, Kappa ACCclass", oa, aa, kappa, each_acc)


cnn.eval()
number = len(TestLabel1)//testSizeNumber
features = []
# pred_y = []
labels = pred_y.data.cpu().numpy()
# print(pred_y)
# print(len(pred_y), np.min(pred_y), np.max(pred_y))
for i in range(number):
    temp = TestPatch1[i * testSizeNumber:(i + 1) * testSizeNumber]
    temp = temp.cuda()
    temp2 = cnn(temp)
    fea = temp2
    features.append(fea.cpu().detach().numpy())
    del temp, temp2

if (i + 1) * testSizeNumber < len(TestLabel1):
    temp = TestPatch1[(i + 1) * testSizeNumber:len(TestLabel1)]
    temp = temp.cuda()
    temp2 = cnn(temp)
    fea = temp2
    features.append(fea.cpu().detach().numpy())
    del temp, temp2


features = np.concatenate(features)
print("features = ", features.shape)



embeddings = TSNE(n_components=2, n_jobs=24).fit_transform(features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
fig = plt.scatter(vis_x, vis_y, c=labels+1, cmap=plt.cm.get_cmap("jet", Classes), marker='.')
plt.colorbar(ticks=range(1,1+Classes))
plt.clim(0.5, 0.5+Classes)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("TSNE_" + datasetName + ".png", dpi=300, format='png', bbox_inches='tight', pad_inches=0)
plt.clf()

