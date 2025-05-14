import numpy as np
import torch
import random
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import net_AMGCFN
import slic
import spectral as spy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []
Seed_List = [1330, 1331, 1332, 1333, 1334,
             1335, 1336, 1337, 1338, 1339]  # 随机种子点
DATASET = 'indian'

# ##数据集预设

if DATASET == 'paviaU':
    data_mat = sio.loadmat('./HyperImage_data/paviaU/PaviaU.mat')
    data = data_mat['paviaU']
    gt_mat = sio.loadmat('./HyperImage_data/paviaU/PaviaU_gt.mat')
    gt = gt_mat['paviaU_gt']
    dataset_name = "paviaU_"  # 数据集名称
    class_count = 9  # 样本类别数
    train_ratio = 0.002  # 训练集比例。注意，训练集为按照‘每类’随机选取
    val_ratio = 0.002  # 验证集比例
elif DATASET == 'indian':
    data_mat = sio.loadmat(
        './HyperImage_data/indian/Indian_pines_corrected.mat')
    data = data_mat['indian_pines_corrected']
    gt_mat = sio.loadmat('./HyperImage_data/indian/Indian_pines_gt.mat')
    gt = gt_mat['indian_pines_gt']
    dataset_name = "indian_"  # 数据集名称
    class_count = 16  # 样本类别数
    train_ratio = 0.02  # 训练集比例。注意，训练集为按照‘每类’随机选取
    val_ratio = 0.02  # 验证集比例
elif DATASET == 'salinas':
    data_mat = sio.loadmat('./HyperImage_data/Salinas/Salinas_corrected.mat')
    data = data_mat['salinas_corrected']
    gt_mat = sio.loadmat('./HyperImage_data/Salinas/Salinas_gt.mat')
    gt = gt_mat['salinas_gt']
    dataset_name = "salinas_"  # 数据集名称
    class_count = 16  # 样本类别数
    train_ratio = 0.002  # 训练集比例。注意，训练集为按照‘每类’随机选取
    val_ratio = 0.002  # 验证集比例

# ##参数预设

sample_type = 'ratio'  # ratio or same
train_samples_per_class = 10  # 当定义为每类样本个数时,则该参数更改为训练样本数
pca_bands = 3
learning_rate = 5e-4  # 学习率
GCN_nhid = 64  # GCN隐藏层通道数
CNN_nhid = 64  # CNN隐藏层通道数
max_epoch = 500  # 迭代次数
superpixel_scale = 100  # 超像素参数
orig_data = data  # 原始数据data
height, width, bands = data.shape  # 原始高光谱数据的三个维度


def GT_To_One_Hot(n_gt, n_class_count):  # 转换为 one-hot 形式
    """
    Convet Gt to one-hot labels
    :param n_gt:
    :param n_class_count:
    :return:
    """
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(n_gt.shape[0]):
        for j in range(n_gt.shape[1]):
            temp = np.zeros(n_class_count, dtype=np.float32)
            if n_gt[i, j] != 0:
                temp[int(n_gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def Draw_Classification_Map_Bg(label, name: str, scale: float = 4.0, dpi: int = 400):  # 绘制有背景的分类图
    """
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    """
    bg_idx = np.where(gt_reshape == 0, 0, 1)
    bg_idx = bg_idx.reshape((height, width))
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel * bg_idx
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale /
                        dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png',
                    transparent=True, dpi=dpi, pad_inches=0)
    plt.show()
    pass


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):  # 绘制全分类图
    """
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    """
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale /
                        dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png',
                    transparent=True, dpi=dpi, pad_inches=0)
    plt.show()
    pass


# Draw_Classification_Map(gt, "results\\" + dataset_name + '_gt')  # 绘制gt

for curr_seed in Seed_List:

    # ##数据预处理

    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    train_data_index = []
    val_rand_idx = []
    rand_idx = []

    # 获取训练集、测试集、验证集
    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'Get train and teat dataset')
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        # print('class ', i, ' ', samplesCount)  # 打印每类数据样本个数
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表

        if sample_type == 'ratio':  # 按每类比例选取
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)  # 训练集样本目录

        elif sample_type == 'same':  # 每类选取相同数量
            real_train_samples_per_class = train_samples_per_class
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)  # 训练集样本目录

        # print('class ', i, ' ', len(rand_idx))  # 输出每类选取样本数量

    # 根据采样目录选取数据组成训练集
    train_rand_idx = np.array(train_rand_idx, dtype=object)
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index)

    # 将测试集（所有样本，包括训练样本）也转化为特定形式
    train_data_index = set(train_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)

    # 背景像元的标签
    background_idx = np.where(gt_reshape == 0)[-1]
    background_idx = set(background_idx)

    # 测试集 = 总数据 - 训练集 - 背景
    test_data_index = all_data_index - train_data_index - background_idx

    # 从测试集中随机选取部分样本作为验证集
    val_data_count = 0
    if sample_type == 'ratio':
        val_data_count = int(
            val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
    elif sample_type == 'same':
        val_data_count = int(class_count)
    val_data_index = random.sample(test_data_index, val_data_count)
    val_data_index = set(val_data_index)

    # 由于验证集为从测试集分裂出，所以测试集应减去验证集
    test_data_index = test_data_index - val_data_index

    # 整理训练集、验证集、测试集
    test_data_index = list(test_data_index)  # 测试集目录
    train_data_index = list(train_data_index)  # 训练集目录
    val_data_index = list(val_data_index)  # 验证集目录

    # 获取训练集的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    # 获取测试集的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    Test_GT = np.reshape(test_samples_gt, [height, width])  # 测试样本图

    # 获取验证集的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
    test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
    val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

    train_samples_gt_onehot = np.reshape(
        train_samples_gt_onehot, [-1, class_count]).astype(int)
    test_samples_gt_onehot = np.reshape(
        test_samples_gt_onehot, [-1, class_count]).astype(int)
    val_samples_gt_onehot = np.reshape(
        val_samples_gt_onehot, [-1, class_count]).astype(int)

    # 制作训练数据和测试数据的GT掩膜,根据GT将带有标签的像元设置为全1向量
    temp_ones = np.ones([class_count])
    # 训练集
    train_label_mask = np.zeros([height * width, class_count])
    train_samples_gt = np.reshape(train_samples_gt, [height * width])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(
        train_label_mask, [height * width, class_count])

    # 测试集
    test_label_mask = np.zeros([height * width, class_count])
    test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(
        test_label_mask, [height * width, class_count])

    # 验证集
    val_label_mask = np.zeros([height * width, class_count])
    val_samples_gt = np.reshape(val_samples_gt, [height * width])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_count])

    # 数据PCA处理 pca_bands = 3
    def pca_process(n_data, n_labels):
        n_labels = np.reshape(n_labels, [-1])
        n_idx = np.where(n_labels != 0)[0]
        x_flatt = np.reshape(n_data, [height * width, bands])
        x = x_flatt[n_idx]
        pca = PCA(n_components=pca_bands)
        pca.fit(x)
        X_new = pca.transform(x_flatt)
        print(pca.explained_variance_ratio_)
        return np.reshape(X_new, [height, width, -1])

    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'PCA processing')
    pca_data = pca_process(data, np.reshape(train_samples_gt, [height, width]))

    # 根据训练集样本进行超像素分割
    superpixel = slic.SlicProcess(pca_data, np.reshape(
        train_samples_gt, [height, width]), class_count - 1)
    tic0 = time.time()
    Q, S, W, Seg = superpixel.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    PCA_SLIC_Time = toc0 - tic0

    print("PCA-SLIC costs time: {}".format(PCA_SLIC_Time))

    # 获取不同hop的图的邻接矩阵
    superpixel_count, _ = W.shape
    pathset = []

    def DFS_get_path(start_node, n_k, get_path):
        if len(get_path) == n_k + 1:
            pathset.append(get_path[:])
        else:
            sub_node = list(np.where(W[start_node] != 0)[0])
            for next_node in sub_node:
                if next_node not in get_path:
                    get_path.append(next_node)
                    DFS_get_path(next_node, n_k, get_path)
                    get_path.pop()
        return 0

    def Get_k_hop(k_hop, n_graph):
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              ' Processing new graph ', k_hop, ' hop')
        new_graph = np.zeros_like(n_graph)
        for center in range(superpixel_count):
            path = [center]
            DFS_get_path(center, k_hop, path)
            for n_path in pathset:
                weight = 0
                for n_node in range(len(n_path) - 1):
                    weight += n_graph[n_path[n_node], n_path[n_node + 1]]
                weight = weight / k_hop
                # if weight > new_graph[n_path[0], n_path[-1]]:
                new_graph[n_path[0], n_path[-1]] = new_graph[n_path[-1], n_path[0]] \
                    = max(weight, new_graph[n_path[0], n_path[-1]], new_graph[n_path[-1], n_path[0]])

            pathset.clear()
        new_graph = new_graph + np.eye(superpixel_count)
        return new_graph

    tic1 = time.time()
    A = Get_k_hop(1, W)
    A2 = Get_k_hop(2, W)
    A3 = Get_k_hop(3, W)
    toc1 = time.time()

    Hop_Graph_Time = toc1 - tic1
    print("Hop-Graph costs time: {}".format(Hop_Graph_Time))
    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'New graph finish')

    def normalization(n_data):
        data_range = np.max(n_data) - np.min(n_data)
        n_data = (n_data - np.min(n_data)) / data_range
        return n_data

    # data = normalization(data)

    # 数据转到GPU运算
    Q = torch.from_numpy(Q).to(device)
    S = torch.from_numpy(S.astype(np.float32)).to(device)

    A = torch.from_numpy(A.astype(np.float32)).to(device)  # 邻接矩阵hop_1
    A2 = torch.from_numpy(A2.astype(np.float32)).to(device)  # 邻接矩阵hop_2
    A3 = torch.from_numpy(A3.astype(np.float32)).to(device)  # 邻接矩阵hop_3

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

    train_samples_gt = torch.from_numpy(
        train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(
        test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(
        val_samples_gt.astype(np.float32)).to(device)

    train_samples_gt_onehot = torch.from_numpy(
        train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(
        test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(
        val_samples_gt_onehot.astype(np.float32)).to(device)

    train_label_mask = torch.from_numpy(
        train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(
        test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(
        val_label_mask.astype(np.float32)).to(device)

    # ##训练网络

    nodes, channel = S.shape
    net = net_AMGCFN.Net(height, width, channel, class_count,
                         GCN_nhid, CNN_nhid, Q, nodes, bands)
    net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=1e-4)
    net.train()
    best_loss = 99999
    best_OA = 0
    train_loss = []
    train_count = []
    train_acc = []

    def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
        real_labels = reallabel_onehot
        we = -torch.mul(real_labels, torch.log(predict))
        we = torch.mul(we, reallabel_mask)
        pool_cross_entropy = torch.sum(we)
        return pool_cross_entropy

    zeros = torch.zeros([height * width]).to(device).float()

    def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                             printFlag=True):
        if not require_AA_KPP:
            with torch.no_grad():
                available_label_idx = (
                    train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                available_label_count = available_label_idx.sum()  # 有效标签的个数
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(
                        train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction.cpu() / available_label_count

                return OA
        else:
            with torch.no_grad():
                # 计算OA
                available_label_idx = (
                    train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                available_label_count = available_label_idx.sum()  # 有效标签的个数
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(
                        train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction.cpu() / available_label_count
                OA = OA.cpu().numpy()

                # 计算AA
                zero_vector = np.zeros([class_count])
                output_data = network_output.cpu().numpy()
                train_samples_gt = train_samples_gt.cpu().numpy()

                output_data = np.reshape(
                    output_data, [height * width, class_count])
                idx = np.argmax(output_data, axis=-1)
                for z in range(output_data.shape[0]):
                    if ~(zero_vector == output_data[z]).all():
                        idx[z] += 1

                count_perclass = np.zeros([class_count])
                correct_perclass = np.zeros([class_count])
                for x in range(len(train_samples_gt)):
                    if train_samples_gt[x] != 0:
                        count_perclass[int(train_samples_gt[x] - 1)] += 1
                        if train_samples_gt[x] == idx[x]:
                            correct_perclass[int(train_samples_gt[x] - 1)] += 1
                test_AC_list = correct_perclass / count_perclass
                test_AA = np.average(test_AC_list)

                # 计算KPP
                test_pre_label_list = []
                test_real_label_list = []
                output_data = np.reshape(
                    output_data, [height * width, class_count])
                idx = np.argmax(output_data, axis=-1)
                idx = np.reshape(idx, [height, width])
                for ii in range(height):
                    for jj in range(width):
                        if Test_GT[ii][jj] != 0:
                            test_pre_label_list.append(idx[ii][jj] + 1)
                            test_real_label_list.append(Test_GT[ii][jj])
                test_pre_label_list = np.array(test_pre_label_list)
                test_real_label_list = np.array(test_real_label_list)
                kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                  test_real_label_list.astype(np.int16))
                test_kpp = kappa

                # 输出
                if printFlag:
                    print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                    print('acc per class:')
                    print(test_AC_list)

                OA_ALL.append(OA)
                AA_ALL.append(test_AA)
                KPP_ALL.append(test_kpp)
                AVG_ALL.append(test_AC_list)

                # 保存数据信息
                f = open('results\\' + dataset_name + '_results.txt', 'a+')
                str_results = '\n======================' \
                              + " learning rate=" + str(learning_rate) \
                              + " epochs=" + str(max_epoch) \
                              + " train ratio=" + str(train_ratio) \
                              + " val ratio=" + str(val_ratio) \
                              + " ======================" \
                              + "\nOA=" + str(OA) \
                              + "\nAA=" + str(test_AA) \
                              + '\nkpp=' + str(test_kpp) \
                              + '\nacc per class:' + str(test_AC_list) + "\n"
                f.write(str_results)
                f.close()
                return OA

    print('Train start')
    tic2 = time.time()
    for epoch in range(max_epoch + 1):
        optimizer.zero_grad()
        out, Cout, Gout = net(S, A, A2, A3, net_input)
        loss = compute_loss(out, train_samples_gt_onehot, train_label_mask)

        Closs = compute_loss(Cout, train_samples_gt_onehot, train_label_mask)
        Gloss = compute_loss(Gout, train_samples_gt_onehot, train_label_mask)
        loss = loss + Closs + Gloss

        train_loss.append(loss.item())
        train_count.append(epoch)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                net.eval()
                out, Cout, Gout = net(S, A, A2, A3, net_input)
                # total = sum([param.nelement() for param in net.parameters()])
                # print('Number of parameter: % .2fK' % (total / 1e3))
                trainloss = compute_loss(
                    out, train_samples_gt_onehot, train_label_mask)
                trainOA = evaluate_performance(
                    out, train_samples_gt, train_samples_gt_onehot)
                valloss = compute_loss(
                    out, val_samples_gt_onehot, val_label_mask)
                valOA = evaluate_performance(
                    out, val_samples_gt, val_samples_gt_onehot)

                print(
                    "{}\ttrain loss={} \t train OA={} \t val loss={} \t val OA={}".format(str(epoch + 1), trainloss,
                                                                                          trainOA, valloss, valOA))
                if valloss < best_loss:
                    best_loss = valloss
                    best_OA = valOA
                    torch.save(net.state_dict(), "model\\best_model.pt")
                    print('save model...')
            torch.cuda.empty_cache()
            net.train()

    toc2 = time.time()
    training_time = toc2 - tic2
    Train_Time_ALL.append(training_time)

    print('Train finished')
    print("Training costs time: {}".format(training_time))
    torch.cuda.empty_cache()

    print('Test start')
    with torch.no_grad():
        net.load_state_dict(torch.load("model\\best_model.pt"))
        net.eval()
        tic3 = time.time()
        out, _, _ = net(S, A, A2, A3, net_input)
        toc3 = time.time()
        testing_time = toc3 - tic3
        # total = sum([param.nelement() for param in net.parameters()])
        # print('Number of parameter: % .2fK' % (total / 1e3))
        testloss = compute_loss(out, test_samples_gt_onehot, test_label_mask)
        testOA = evaluate_performance(out, test_samples_gt, test_samples_gt_onehot,
                                      require_AA_KPP=True, printFlag=False)
        print("out test loss={}\t test OA={} ".format(testloss, testOA))
        print('OA=', np.array(OA_ALL))
        classification_map = torch.argmax(
            out, 1).reshape([height, width]).cpu() + 1
        Draw_Classification_Map(
            classification_map, "results\\" + dataset_name + str(testOA))
        Draw_Classification_Map_Bg(
            classification_map, "results\\" + dataset_name + str(testOA) + "Bg")
        Test_Time_ALL.append(testing_time)
    print('Test finished')
    torch.cuda.empty_cache()
    del net


OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

print(DATASET)
print("\ntrain_ratio={}".format(train_ratio),
      "\n==============================================================================")
print('OA=', OA_ALL)
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
print("Average training time:{}".format(np.mean(Train_Time_ALL)))
print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

# 保存数据信息
f = open('results\\' + dataset_name + '_results.txt', 'a+')
str_results = '\n\n************************************************' \
              + "\ntrain_ratio={}".format(train_ratio) \
              + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
              + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
              + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
              + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
              + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
              + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
f.write(str_results)
f.close()
