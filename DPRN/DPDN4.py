import argparse
import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,cohen_kappa_score
from operator import truediv
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.parallel
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def random_unison(a,b,rstate=None):
    assert len(a)==len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p],b[p]
# split by samples per class
def split_data_fix(pixels,labels,n_samples,rand_state=None):
    pixels_number = np.unique(labels,return_counts=1)[1]
    train_set_size = [n_samples] * len(np.unique(labels))
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    sizetr = np.array([tr_size]+list(pixels.shape)[1:])
    sizete = np.array([te_size]+list(pixels.shape)[1:])
    train_x = np.empty((sizetr));
    train_y = np.empty((tr_size));
    test_x = np.empty((sizete));
    test_y = np.empty((te_size))
    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        pixels_cl = pixels[labels==cl]
        labels_cl = labels[labels==cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont,:,:,:] = a
                train_y[trcont] = b
                trcont += 1
            else:
                test_x[tecont,:,:,:] = a
                test_y[tecont] = b
                tecont += 1
    train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
    return train_x, test_x, train_y, test_y
# split by percent
def split_data(pixels,labels,percent,splitdset="custom",rand_state=69):
    splitdset = "sklearn"
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1-percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size]+list(pixels.shape)[1:])
        sizete = np.array([te_size]+list(pixels.shape)[1:])
        train_x = np.empty((sizetr));
        train_y = np.empty((tr_size));
        test_x = np.empty((sizete));
        test_y = np.empty((te_size))
        trcont = 0;
        tecont = 0;
        for cl in np.unique(labels):
            pixels_cl = pixels[labels==cl]
            labels_cl = labels[labels==cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont,:,:,:] = a
                    train_y[trcont] = b
                    trcont += 1
                else:
                    test_x[tecont,:,:,:] = a
                    test_y[tecont] = b
                    tecont += 1
        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
        return train_x, test_x, train_y, test_y

def loadData(name,num_components=None):
    data_path = os.path.join(os.getcwd(),'./data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path,'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    else:
        print("NO DATASET")
        exit()
    shapeor = data.shape
    data = data.reshape(-1,data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels))-1
    return data,labels,num_class

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int")

def accuracy(output,target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    """ .topK() 用来求tensor中某个dim的前k大或者前k小的值以及对应的index"""
    _,pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, y_test, name):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))

class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]
    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)
    def __labels__(self):
        return self.labels


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)#kernel_size=3
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x,out),1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,padding=1, bias=False)#kernel_size=3
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class pDenseNet(nn.Module):
    def __init__(self, growthRate,depth,reduction,nClasses,bottleneck,avgpoosize):
        super(pDenseNet, self).__init__()
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(176, nChannels, kernel_size=3, padding=1,bias=False)#kernel_size=3,
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.avgpool = nn.AvgPool2d(2)#avgpoosize)
        self.fc = nn.Linear(nChannels,nClasses)#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
    def forward(self, x):
        print("X size:",x.size())
        out = self.conv1(x)
        print("after conv1 out size",out.size())
        out = self.trans1(self.dense1(out))
        print("after dense1+trans1 out size",out.size())
        out = self.trans2(self.dense2(out))
        print("after dense2+trans2 out size",out.size())
        out = self.trans3(self.dense3(out))
        print("after dense3+trans3 out size",out.size())
        out = self.dense4(out)
        #out = self.dense3(out)
        print("after dense4 out size",out.size())
        out = self.bn1(out)
        print("after bn1 out size",out.size())
        out = F.relu(out)
        print("after relu out size",out.size())
        #out = F.avg_pool2d(out,1)
        out = self.avgpool(out)
        print("after avg_pool out size",out.size())
        out = torch.squeeze(out)# 8))
        print("after squeeze out size",out.size())
        out = self.fc(out)
        print("after fc out size",out.size())
        out = F.log_softmax(out)
        return out

def load_hyper(args):
    data, labels, numclass = loadData(args.dataset, num_components=args.components)
    pixels, labels = createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = True)
    bands = pixels.shape[-1];
    numberofclass = len(np.unique(labels))
    if args.tr_percent < 1: # split by percent
        x_train, x_test, y_train, y_test = split_data(pixels, labels, args.tr_percent)
    else: # split by samples per class
        x_train, x_test, y_train, y_test = split_data_fix(pixels, labels, args.tr_percent)
    if args.use_val:
        x_val, x_test, y_val, y_test = split_data(x_test, y_test, args.val_percent)
    del pixels, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train))
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test))
    if args.use_val:
        val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val))
    else:
        val_hyper = None
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, numberofclass, bands

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs   = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # print('inputs=', inputs.size())
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))

def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        losses[batch_idx] = criterion(outputs, targets).item()
        accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))

def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='KSC', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.1, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=1000, type=int, help='mini-batch test size (default: 1000)')
    parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--alpha', default=48, type=int, help='number of new channel increases per depth (default: 12)')
    parser.add_argument('--inplanes', dest='inplanes', default=21, type=int, help='bands before blocks')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock (default: bottleneck)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=27, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.set_defaults(bottleneck=True)
    best_err1 = 100
    #args = parser.parse_args()
    args =parser.parse_known_args()[0]
    state = {k: v for k, v in args._get_kwargs()}
    train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(args)
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True
    if args.spatialsize < 9: avgpoosize = 1
    elif args.spatialsize <= 11: avgpoosize = 2
    elif args.spatialsize == 15: avgpoosize = 3
    elif args.spatialsize == 19: avgpoosize = 4
    elif args.spatialsize == 21: avgpoosize = 5
    elif args.spatialsize == 27: avgpoosize = 6
    elif args.spatialsize == 29: avgpoosize = 7
    else: print("spatialsize no tested")
    '''
    pDenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck)
    '''
    model = pDenseNet(growthRate=args.inplanes,depth=args.depth,reduction=0.5,nClasses=num_classes,
                      bottleneck=args.bottleneck,avgpoosize=avgpoosize)
    #model = pDenseNet(args.depth, args.alpha, num_classes, n_bands, avgpoosize, args.inplanes, bottleneck=args.bottleneck)
    if use_cuda: model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    best_acc = -1
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        if args.use_val: test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        else: test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
        print("EPOCH", epoch, "TRAIN LOSS", train_loss, "TRAIN ACCURACY", train_acc, end=',')
        print("LOSS", test_loss, "ACCURACY", test_acc)
        # save model
        if test_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")
            best_acc = test_acc
    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    classification, confusion, results = reports(np.argmax(predict(test_loader, model, criterion, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()), args.dataset)
    print(classification)
    print(args.dataset, results)

if __name__ == '__main__':
    main()
