import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm


def DrawResult(labels, imageID):
    # ID=1:Pavia University
    # ID=2:Salinas
    # ID=3:Houston2013
    # ID=4:Indian_pines
    # ID=5:LongKou
    # ID=6:HanChuan
    # ID=7:HongHu
    # ID=8:Houston2018
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255

    elif imageID == 2:
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette * 1.0 / 255

    elif imageID == 3:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette * 1.0 / 255

    elif imageID == 4:
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255

    elif imageID == 5:
        row = 550
        col = 400
        palette = np.array([[255, 0, 0],
                            [239, 155, 0],
                            [255, 255, 0],
                            [0, 255, 0],
                            [0, 255, 255],
                            [0, 140, 140],
                            [0, 0, 255],
                            [255, 255, 255],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255

    elif imageID == 6:
        row = 1217
        col = 303
        palette = np.array([[176, 48, 96],
                            [0, 255, 255],
                            [255, 0, 255],
                            [160, 32, 240],
                            [127, 255, 212],
                            [127, 255, 0],
                            [0, 205, 0],
                            [0, 255, 0],
                            [0, 139, 0],
                            [255, 0, 0],
                            [216, 191, 216],
                            [255, 127, 80],
                            [160, 82, 45],
                            [255, 255, 255],
                            [218, 112, 214],
                            [0, 0, 255],
                            ])
        palette = palette * 1.0 / 255
    elif imageID == 7:
        row = 940
        col = 475
        palette = np.array([[255, 0, 0],
                            [255, 255, 255],
                            [176, 48, 96],
                            [255, 255, 0],
                            [255, 127, 80],
                            [0, 255, 0],
                            [0, 205, 0],
                            [0, 139, 0],
                            [127, 255, 212],
                            [160, 32, 240],
                            [216, 191, 216],
                            [0, 0, 255],
                            [0, 0, 139],
                            [218, 112, 214],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 165, 0],
                            [127, 255, 0],
                            [139, 139, 0],
                            [0, 139, 139],
                            [205, 181, 205],
                            [238, 154, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 8:
        row = 601
        col = 2384
        palette = np.array([[0, 206, 0],
                            [123, 220, 0],
                            [47, 139, 85],
                            [0, 136, 0],
                            [0, 68, 0],
                            [159, 79, 41],
                            [68, 227, 251],
                            [255, 255, 255],
                            [213, 191, 213],
                            [248, 2, 0],
                            [167, 160, 146],
                            [124, 124, 124],
                            [160, 4, 3],
                            [80, 0, 5],
                            [226, 161, 13],
                            [255, 242, 3],
                            [237, 153, 0],
                            [242, 0, 200],
                            [0, 6, 191],
                            [172, 196, 219]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 10:
        row = 1740
        col = 860
        palette = np.array([[140, 67, 46],
                            [153, 153, 153],
                            [255, 100, 0],
                            [0, 255, 123],
                            [164, 75, 155],
                            [101, 174, 255],
                            [118, 254, 172],
                            [60, 91, 112],
                            [255, 255, 0],
                            [255, 255, 125],
                            [255, 0, 255],
                            [100, 0, 255],
                            [0, 172, 254],
                            [0, 255, 0],
                            [171, 175, 80],
                            [101, 193, 60],
                            [139, 0, 0],
                            [0, 0, 255]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 11:
        row = 880
        col = 1360
        palette = np.array([[0, 255, 0],
                            [153, 153, 153],
                            [255, 100, 0],
                            [164, 75, 155],
                            [101, 174, 255],
                            [140, 67, 46]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 12:
        row = 1230
        col = 1000
        palette = np.array([[140, 67, 46],
                            [0, 0, 255],
                            [0, 200, 0],
                            [101, 174, 255],
                            [164, 75, 155],
                            [192, 80, 70],
                            [60, 91, 112],
                            [255, 255, 0],
                            [255, 100, 0],
                            [118, 254, 172]
                            ])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(1, num_class + 1):
        X_result[np.where(labels == i), 0] = palette[i - 1, 0]
        X_result[np.where(labels == i), 1] = palette[i - 1, 1]
        X_result[np.where(labels == i), 2] = palette[i - 1, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA


def LoadHSI(dataID=1, num_label=150):
    # ID=1:Pavia University
    # ID=2:Salinas
    # ID=3:Houston2013
    # ID=4:Indian_pines
    # ID=5:LongKou
    # ID=6:HanChuan
    # ID=7:HongHu
    # ID=8:Houston2018

    if dataID == 1:
        data = sio.loadmat('./Data/PaviaU.mat')
        X = data['paviaU']
        data = sio.loadmat('./Data/PaviaU_gt.mat')
        Y = data['paviaU_gt']

    elif dataID == 2:
        data = sio.loadmat('./Data/Salinas_corrected.mat')
        X = data['salinas_corrected']
        data = sio.loadmat('./Data/Salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID == 3:
        data = sio.loadmat('./Data/GRSS2013.mat')
        X = data['GRSS2013']
        data = sio.loadmat('./Data/GRSS2013_gt.mat')
        Y = data['GRSS2013_gt']

    elif dataID == 4:
        data = sio.loadmat('./Data/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('./Data/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        # num_label = [30, 50, 50, 50, 50, 50, 20, 50, 15, 50, 50, 50, 50, 30, 50, 50]

    elif dataID == 5:
        data = sio.loadmat('./Data/WHU_Hi_LongKou.mat')
        X = data['WHU_Hi_LongKou']
        data = sio.loadmat('./Data/WHU_Hi_LongKou_gt.mat')
        Y = data['WHU_Hi_LongKou_gt']

    elif dataID == 6:
        data = sio.loadmat('./Data/WHU_Hi_HanChuan.mat')
        X = data['WHU_Hi_HanChuan']
        data = sio.loadmat('./Data/WHU_Hi_HanChuan_gt.mat')
        Y = data['WHU_Hi_HanChuan_gt']

    elif dataID == 7:
        data = sio.loadmat('./Data/WHU_Hi_HongHu.mat')
        X = data['WHU_Hi_HongHu']
        data = sio.loadmat('./Data/WHU_Hi_HongHu_gt.mat')
        Y = data['WHU_Hi_HongHu_gt']

    elif dataID == 8:
        data = sio.loadmat('./Data/Houston2018.mat')
        X = data['Houston2018']
        data = sio.loadmat('./Data/Houston2018_gt.mat')
        Y = data['Houston2018_gt']

    [row, col, n_feature] = X.shape
    K = row * col
    X = X.reshape(K, n_feature)

    n_class = Y.max()

    X = featureNormalize(X, 2)
    X = np.reshape(X, (row, col, n_feature))
    X = np.moveaxis(X, -1, 0)
    Y = Y.reshape(K, ).astype('int')

    for i in range(1, n_class + 1):

        index = np.where(Y == i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        train_num = num_label
        # train_num = num_label[i-1]
        if i == 1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:
            train_array = np.append(train_array, index[randomArray_label[0:train_num]])
            test_array = np.append(test_array, index[randomArray_label[train_num:n_data]])

    return X, Y, train_array, test_array