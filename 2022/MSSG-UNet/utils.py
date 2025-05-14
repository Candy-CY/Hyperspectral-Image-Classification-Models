import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import random
def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def distcorr(X, Y):
    """ Compute the distance correlation function
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    # print(X.shape, Y.shape)
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def applyPCA(X, n_components=3):
    # newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=n_components, whiten=True)
    newX = pca.fit_transform(X)
    # newX = np.reshape(newX, (X.shape[0], X.shape[1], n_components))
    return newX


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt: 2D
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    [height, width]=gt.shape
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def get_Samples_GT(seed: int, gt: np.array, class_count: int, train_ratio,val_ratio, samples_type: str = 'ratio', ):
    # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        train_number_per_class=[]
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32')+\
                                     np.ceil(samplesCount*val_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        val_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = list(train_rand_idx[c])
            train_data_index=train_data_index+a[:train_number_per_class[c]]
            val_data_index=val_data_index+a[train_number_per_class[c]:]
        
        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        val_data_index=set(val_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        test_data_index = all_data_index - train_data_index - val_data_index
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    if samples_type == 'same_num':
        if int(train_ratio) == 0 or int(val_ratio) == 0:
            print("ERROR: The number of samples for train. or val. is equal to 0.")
            exit(-1)
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio) # 每类相同数量样本,则训练比例为每类样本数量
            real_val_samples_per_class = int(val_ratio) # 每类相同数量样本,则训练比例为每类样本数量
            
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class >= samplesCount:
                real_train_samples_per_class = samplesCount-1
                real_val_samples_per_class=1
            else: real_val_samples_per_class= real_val_samples_per_class if (real_val_samples_per_class+real_train_samples_per_class)<=samplesCount else samplesCount-real_train_samples_per_class
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class+real_val_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
            if real_val_samples_per_class>0:
                rand_real_idx_per_class_val = idx[rand_idx[-real_val_samples_per_class:]]
                val_rand_idx.append(rand_real_idx_per_class_val)


        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        
        val_data_index = []
        for c in range(val_rand_idx.shape[0]):
            a = val_rand_idx[c]
            for j in range(a.shape[0]):
                val_data_index.append(a[j])
        val_data_index = np.array(val_data_index)
        
        
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        test_data_index = all_data_index - train_data_index - val_data_index
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    
    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    
    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass
    
    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    return train_samples_gt, test_samples_gt, val_samples_gt