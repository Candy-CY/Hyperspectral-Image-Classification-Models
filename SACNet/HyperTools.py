import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py

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
        
def DrawResult(labels,imageID):
    num_class = int(labels.max())
    if imageID == 1:#ID=1:Pavia University 9个类
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    
    elif imageID ==2:#ID=2:Salinas 16个类
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
        palette = palette*1.0/255

    elif imageID ==3: #Indian Pines 16个类
        row = 145
        col = 145
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

    elif imageID ==4: #KSC  13个类
        row = 512
        col = 614
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
                            [239, 71, 34]])

    elif imageID ==5: #xuzhou 9个类
        row = 500
        col = 260
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78]])

    elif imageID ==6: #HoustonU 15个类
        row = 349
        col = 1905
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
                            [180, 31, 35]])

    elif imageID ==7: #xiongan 19个类
        row = 3750
        col = 1580
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
                            [123, 18, 20],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])

    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
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

def LoadHSI(dataID=1,num_label=150):
    # /media/kris-allen/新加卷1/HSI
    #ID=1:PU
    #ID=2:SV
    #ID=3: IP
    #ID=4: KSC
    #ID=5: xuzhou
    #ID=6:Houston
    #ID=7:xiong'an
    if dataID==1:        
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/paviaU.mat')
        X = data['paviaU']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/paviaU_gt.mat')
        Y = data['paviaU_gt']

    elif dataID==2:        
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/salinas_corrected.mat')
        X = data['salinas_corrected']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID==3:
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']

    elif dataID==4:
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/KSC.mat')
        X = data['KSC']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/KSC_gt.mat')
        Y = data['KSC_gt']

    elif dataID==5:
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/HSI_Data/Xuzhou/xuzhou.mat')
        X = data['xuzhou']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/HSI_Data/Xuzhou/xuzhou_gt.mat')
        Y = data['xuzhou_gt']

    elif dataID==6:
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/HSI_Data/HoustonU/Houston.mat')
        X = data['Houston']
        data = sio.loadmat('/media/kris-allen/新加卷1/HSI/HSI_Data/HoustonU/Houston_gt.mat')
        Y = data['Houston_gt']

    elif dataID==7:
        #data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/xiongan.mat')
        data = h5py.File('/media/kris-allen/新加卷1/HSI/data/xiongan.mat')
        #print(data.keys())
        X = data['XiongAn']
        X = np.transpose(X)
        #data = sio.loadmat('/media/kris-allen/新加卷1/HSI/data/xiongan_gt.mat')
        data = h5py.File('/media/kris-allen/新加卷1/HSI/data/xiongan_gt.mat')
        Y = data['xiongan_gt']
        Y = np.transpose(Y)

    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(K, n_feature)
    n_class = Y.max()
    X = featureNormalize(X,2)  
    X = np.reshape(X,(row,col,n_feature))
    X = np.moveaxis(X,-1,0)
    Y = Y.reshape(K,).astype('int')

    for i in range(1,n_class+1):
        
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        train_num = num_label
        if i==1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:            
            train_array = np.append(train_array,index[randomArray_label[0:train_num]])
            test_array = np.append(test_array,index[randomArray_label[train_num:n_data]])

    return X,Y,train_array,test_array