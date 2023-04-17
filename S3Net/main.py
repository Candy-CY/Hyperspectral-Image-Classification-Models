import torch
from torchvision import transforms
import random
import numpy as np
from loss_function import ContrastiveLoss
from model import S3Net
import copy
from Function import DrawCluster,DrawResult, get_k_layer_feature_map, AA_andEachClassAccuracy, information_process, show_feature_map
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
import argparse
from data_read import get_data
apex = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#存在特征融合，fusion的步骤
def train(encoder,Datapath1,Datapath2,PairLabelpath, X_train, y_train,trans, windowsize1, windowsize2, margin, epochs=50):

    optim_contrast = torch.optim.Adam(encoder.parameters(), lr=5e-5, weight_decay=0.00000)
    contrast_criterion = ContrastiveLoss(margin)

    optim_classi = torch.optim.Adam(encoder.parameters(), lr=args.classi_lr, weight_decay=0.00005)
    criterion = torch.nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(encoder.state_dict())

    epoch_contrastloss = 0
    best_loss = 10000
    best_acc = 0
    for epoch in range(epochs):
        contrast_data = random_PairSelect(Datapath1, Datapath2, PairLabelpath, trans, windowsize1, windowsize2)
        contrast_loader = torch.utils.data.DataLoader(dataset=contrast_data, batch_size=args.batch_size, shuffle=True,
                                                      drop_last=False)
        classi_data = ClassiDataset(X_train, y_train, trans, windowsize1, windowsize2, 'Train')
        classi_loader = torch.utils.data.DataLoader(dataset=classi_data, batch_size=args.batch_size, shuffle=True)

        train_acc = 0
        epoch_classiloss=0
        print(" Epoch No  {} ".format(epoch))
        #contrastive_learning
        for step, (Data_1, Data_2, label) in enumerate(contrast_loader):
            Data_i = Data_1.cuda().float()
            Data_j = Data_2.cuda().float()
            label = label.cuda().float()
            encoder = encoder.cuda()
            encoder.train()
            z_i, z_j, z_i_, z_j_ = encoder(Data_i, Data_j)
            contrast_loss = contrast_criterion(z_i, z_j, label, z_i_, z_j_)
            epoch_contrastloss += contrast_loss.item()
            optim_contrast.zero_grad()
            contrast_loss.backward(retain_graph=True)
            optim_contrast.step()
        #classified learning
        for i, (Data_1, Data_2, label) in enumerate(classi_loader):
            Data_1 = Data_1.cuda().float()
            Data_2 = Data_2.cuda().float()
            label-=1
            label = label.cuda()
            encoder = encoder.cuda()
            encoder.train()
            z_i, z_j, z_i_, z_j_ = encoder(Data_1, Data_2)
            classi_loss = criterion(z_i+z_j, label.long())
            epoch_classiloss = epoch_classiloss + classi_loss.item()
            pred = torch.max(z_i+z_j, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optim_classi.zero_grad()
            classi_loss.backward(retain_graph=True)
            optim_classi.step()
        print(
            'Train Loss: {:.6f}, Acc: {:.6f}, Contrast Loss: {:.6f}'.format(classi_loss / (len(classi_data)), train_acc / (len(classi_data)),contrast_loss / (len(contrast_data))))
        if (train_acc / (len(classi_data)) >= best_acc) and ((classi_loss / (len(classi_data))+contrast_loss / (len(contrast_data))) < best_loss):
            best_loss=(classi_loss / (len(classi_data))+contrast_loss / (len(contrast_data)))
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc=train_acc / (len(classi_data))
    class_feature = []
    torch.save(best_model_wts, modelpath_front + 'model.pth')
    return class_feature
def predict(model,x_test_idx, y_test, trans, windowsize1, windowsize2):
    model.eval()
    with torch.no_grad():
        model = model.cuda()
        test_data = ClassiDataset(x_test_idx, y_test, trans, windowsize1, windowsize2, 'Test')
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        prediction = []
        metric = []

        for data_1, data_2, label in test_loader:
            data_1=data_1.cuda().float()
            data_2=data_2.cuda().float()
            z_i, z_j, z_i_, z_j_ = model(data_1, data_2)
            out = z_i + z_j
            for num in range(len(out)):
                prediction.append(np.array(out[num].cpu().detach().numpy()))
    return prediction, metric
def reports(model,x_test_idx, y_test, name, trans, windowsize1, windowsize2):
    y_pred_index, metric = predict(model,x_test_idx, y_test, trans, windowsize1, windowsize2)
    y_pred = np.argmax(np.array(y_pred_index), axis=1)
    Label=y_test
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'HS':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
            , 'Soil', 'Water', 'Residential',
                        'Commercial', 'Road', 'Highway', 'Railway',
                        'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
    elif name == 'PCR' or name == 'PCL':
        target_names = ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks'
            , 'Bitumen', 'Tiles', 'Shadows',
                        'Meadows', 'Bare Soil']
    elif name == 'PD':
        target_names = ['Road', 'Grass', 'Shadow', 'Soil', 'Tree', 'Roof']
    elif name == 'HHK':
        target_names = ['Reed', 'Spartina alterniflora', 'Salt filter pond', 'Salt evaporation pond',
           'Dry pond', 'Tamarisk', 'Salt pan', 'Seepweed', 'River', 'Sea', 'Mudbank', 'Tidal creek',
           'Fallow land', 'Ecological restoration pond', 'Robinia', 'Fishpond', 'Pit pond',
           'Building', 'Bare land', 'Paddyfield', 'Cotton', 'Soybean', 'Corn']
    Label = np.array(Label)-1
    Label = list(Label)
    classification = classification_report(Label, y_pred, target_names=target_names)
    oa = accuracy_score(Label, y_pred)
    # DrawCluster(Label, y_pred_index, oa)
    confusion = confusion_matrix(Label, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(Label, y_pred)
    print('OA:', oa)
    print(confusion)
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100, y_pred, metric
class random_PairSelect(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath1,Datapath2,Labelpath, trans, windowsize1, windowsize2):
        # 1. Initialize file path or list of file names.
        self.data = X
        self.data_idx = x_train_idx
        self.output_units = output_units#类别
        self.perclass = perclass
        self.DataList1 = Datapath1
        self.DataList2 = Datapath2
        self.LabelList = Labelpath
        self.trans=trans
        self.windowsize1 = windowsize1
        self.windowsize2 = windowsize2
        un_matchlist=[i for i in range(len(self.LabelList)) if self.LabelList[i] == 0]
        matchlist = [i for i in range(len(self.LabelList)) if self.LabelList[i] == 1]

        self.un_matchlist = random.sample(un_matchlist, int(self.output_units*self.perclass))#2
        self.matchlist = random.sample(matchlist, int(self.output_units*self.perclass))#1
        self.un_matchlist.extend(self.matchlist)
        random.shuffle(self.un_matchlist)
    def neighbor_add(self, row, col, ws, w_size=3):  # 给出 row，col和标签，返回w_size大小的cube
        res = np.zeros([w_size*2+1, w_size*2+1, self.data.shape[2]])
        res[:] = self.data[row+ws-w_size:row + ws+ w_size+1, col+ws-w_size:col + ws+ w_size+1, :]
        return res
    def __getitem__(self, index):
        list_index=self.un_matchlist[index]
        num1=self.DataList1[list_index]
        idx = self.data_idx[num1]
        data1 = self.neighbor_add(idx[0], idx[1], self.windowsize1//2, self.windowsize1//2)
        Data_1=self.trans(data1.astype('float64'))
        Data_1=Data_1.view(-1,Data_1.shape[0],Data_1.shape[1],Data_1.shape[2])
        num2=self.DataList2[list_index]
        idx = self.data_idx[num2]
        data2 = self.neighbor_add(idx[0], idx[1], self.windowsize1//2, self.windowsize2//2)
        Data_2=self.trans(data2.astype('float64'))
        Data_2=Data_2.view(-1,Data_2.shape[0],Data_2.shape[1],Data_2.shape[2])
        Label=self.LabelList[list_index]
        return Data_1, Data_2, Label
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # return len(self.DataList1)
        if perclass>=1:return int(2*self.output_units*self.perclass)
        else:return 2*self.output_units*self.perclass
class ClassiDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self, x_train_idx, y_train, transform, w_size1, w_size2, model = 'Train'):
        # 1. Initialize file path or list of file names.
        self.data = X
        self.data_idx = x_train_idx
        self.ws1 = w_size1
        self.ws2 = w_size2
        self.Labellist=y_train
        self.model = model
        self.transform=transform
        # self.neiborindex=neiborindex
    def neighbor_add(self, row, col, ws, w_size=3):  # 给出 row，col和标签，返回w_size大小的cube
        res = np.zeros([w_size*2+1, w_size*2+1, self.data.shape[2]])
        res[:] = self.data[row+ws-w_size:row + ws+ w_size+1, col+ws-w_size:col + ws+ w_size+1, :]
        return res
    def __getitem__(self, index):
        index=index
        data1 = self.neighbor_add(self.data_idx[index][0], self.data_idx[index][1], self.ws1//2, self.ws1//2)
        data2 = self.neighbor_add(self.data_idx[index][0], self.data_idx[index][1], self.ws1//2, self.ws2//2)
        Data_1=self.transform(data1.astype('float64'))
        Data_2=self.transform(data2.astype('float64'))
        Data_1=Data_1.view(-1,Data_1.shape[0],Data_1.shape[1],Data_1.shape[2])
        Data_2=Data_2.view(-1,Data_2.shape[0],Data_2.shape[1],Data_2.shape[2])
        if self.model == 'Train':
            label = self.Labellist[index]
        else:
            label = 0
        return Data_1, Data_2, label
    def __len__(self):

        return len(self.data_idx)
dataset_names = ['IP', 'SA', 'PU', 'PCL','PCR','HHK','PD']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='SA', choices=dataset_names)
parser.add_argument('--train',type=bool, default=1)
parser.add_argument('--perclass', type=float, default=20)
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0"))
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--windowSize1', type=int, default=7)#1 is big patch
parser.add_argument('--windowSize2', type=int, default=3)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--classi_lr', type=float, default=5e-3)
parser.add_argument('--draw', type=bool, default=True)
parser.add_argument('--test_batch_size', type=int, default=99)
parser.add_argument('--output_units', type=int, default=16)
parser.add_argument('--pca', type=bool, default=True)
parser.add_argument('--margin', type=float, default=1.2)
args = parser.parse_args()
TRAIN =args.train
epoch=args.epoch
windowSize1 = args.windowSize1
windowSize2 = args.windowSize2
dataset = args.dataset
pca = args.pca
margin = args.margin
perclass = args.perclass
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]#To ensure that the samples of 10 times are the same, we use a seed list which comes from DCFSL https://github.com/Li-ZK/DCFSL-2021/
if dataset == 'PU' or dataset == 'PCR' or dataset == 'PCL':
    output_units = 9
elif dataset == 'IP' or dataset == 'SA':
    output_units = 16
elif dataset == 'HS':
    output_units = 15
elif dataset == 'HHK':
    output_units = 23
if pca == True:
    if dataset == 'PU' or dataset == 'PCR' or dataset == 'PCL' :
        K = 30
    elif dataset == 'SA' or dataset == 'IP':
        K = 60
    elif dataset == 'HHK':
        K = 80
else:
    K = 200 if dataset == 'IP' else 103


# IP 10249   PU 42776   SA  54129
trans = transforms.Compose(transforms=[
    transforms.ToTensor(),
    transforms.Normalize(np.zeros(K), np.ones(K))])
add_info = []
Each_class_acc = []
iteration = 10
for i in range(iteration):
    modelpath_front = './model_result/' + dataset + '_' + str(i+1) + '_' + str(perclass) + 'sample'
    print('time:', i+1,'windowSize1:', windowSize1, 'windowSize2', windowSize2, 'dataset:', dataset)
    model=S3Net(channel=K,output_units=output_units)
    model=model.cuda()
    print('dataset:', dataset)
    sed = seeds[i]
    if TRAIN:
        print('data_load_start')
        dataload_start_time = time.time()
        X, x_train_idx, y_train, x_test_idx, y_test, datalist1, datalist2, labellist, whole_idx = get_data(dataset=dataset,windowSize1=windowSize1,perclass=perclass, seed=sed, K=K, PCA=pca)
        print('data_load_over', 'usetime:', time.time() - dataload_start_time)
        train_start_time = time.time()
        class_feature = train(model,datalist1,datalist2,labellist,
                                     x_train_idx, y_train ,trans, windowSize1, windowSize2, margin=margin,epochs=epoch)
        print('train_time:', time.time()-train_start_time)
    #Testing
    model.load_state_dict(torch.load(modelpath_front + 'model.pth'))
    test_start_time=time.time()

    classification, confusion, oa, each_acc, aa, kappa, y_pred, metric_pred = reports(model,x_test_idx, y_test, dataset, trans, windowSize1, windowSize2)
    print('train_time:', time.time() - test_start_time)
    # if args.draw:
    #     X_RES = DrawResult(metric_pred, dataset, oa)

    print('testing time:',time.time()-test_start_time,'s')
    add_info.append([oa,kappa,aa])
    Each_class_acc.append(each_acc)
information_process(dataset, windowSize1, windowSize2, perclass, args.batch_size, iteration, K, add_info, np.array(Each_class_acc), margin)
