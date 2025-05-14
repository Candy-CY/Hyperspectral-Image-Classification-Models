import scipy.io as scio
import numpy as np
import random
from utils import convert_to_one_hot

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def Mat_dis(x):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)   #构建矩阵
    aa = np.sum(np.multiply(x, x), 1)   #哈达玛乘积
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    #dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat


def Mat_dis_s2(x, sig=0.0001, alp = 0.5):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    spe = x[:, :-2]
    spa = x[:, -2:]
    dist_spetral = Mat_dis(spe) / spe.shape[1]
    dist_spatial = Mat_dis(spa) / spa.shape[1]
    # dist_spetral = np.exp(-sig*dist_spetral)
    # dist_spatial = np.exp(-sig*dist_spatial)
    #dist_mat = alp*dist_spetral + (1-alp)*dist_spatial


    return dist_spetral, dist_spatial

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, sig=1000):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    A = np.mean(dis_mat)
    print(A)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 1.0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        #avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(- sig * dis_vec[0, node_idx] / A )
                #print(H[node_idx, center_idx])
                #H[node_idx, center_idx] = dis_vec[0, node_idx]
            else:
                H[node_idx, center_idx] = 1.0
    return H

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        W = np.mat(np.diag(W))
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_Graph(H):
    """
    generate graph G with H
    :param H:
    :return:
    """
    H = np.maximum(H,H.T)
    D = np.sum(H, axis=1)
    D2 = np.mat(np.diag(np.power(D, -0.5)))
    G = D2 * H * D2

    return G

def data_prepare_whole(num_class, variable_weight = False, k_spe = 16, k_spa = 16):
    """

    :param num_class:
    :param variable_weight:
    :param k_spe: hyperparameter
    :param k_spa: hyperparameter
    :return: prepared data for training
    """
    # choose dataset
    img = scio.loadmat('./Datasets/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = scio.loadmat('./Datasets/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
    # img = scio.loadmat('./Datasets/KSC/KSC.mat')['KSC']
    # gt = scio.loadmat('./Datasets/KSC/KSC_gt.mat')['KSC_gt']
    # img = scio.loadmat('./Datasets/Botswana/Botswana.mat')['Botswana']
    # gt = scio.loadmat('./Datasets/Botswana/Botswana_gt.mat')['Botswana_gt']
    # Max = np.max(img)
    # Min = np.min(img)
    # img = (img - Min) / (Max - Min)

    h, w = gt.shape[0], gt.shape[1]
    c = img.shape[2]
    idx = np.ones((img.shape[0], img.shape[1]))
    idx = np.where(idx == 1)
    idx_x = np.resize(idx[0], (img.shape[0], img.shape[1], 1))
    idx_y = np.resize(idx[1], (img.shape[0], img.shape[1], 1))
    img_idx = np.concatenate((idx_x, idx_y), axis=2)
    # img_with_idx = np.concatenate((img, idx_x), axis=2)
    # img_with_idx = np.concatenate((img_with_idx, idx_y), axis=2)
    img_with_idx = np.concatenate((img, img_idx), axis=2)
    gt_train = np.zeros((img.shape[0], img.shape[1]))
    # choose num of training samples
    for i in range(1, num_class + 1):
        id = np.where(gt == i)
        num = id[0].shape[0]
        if num >= 50:
            a = np.ones(num)
            a = np.where(a == 1)
            a = list(a[0])
            idx_rand = random.sample(a, 50)

        else:
            a = np.ones(num)
            a = np.where(a == 1)
            a = list(a[0])
            idx_rand = random.sample(a, 15)
        for item in idx_rand:
            x = id[0][item]
            y = id[1][item]
            gt_train[x][y] = i
    gt_test = gt - gt_train
    gt_train = np.resize(gt_train, (gt_train.shape[0]*gt_train.shape[1]))
    gt_test = np.resize(gt_test, (gt_test.shape[0]*gt_test.shape[1]))
    img_with_idx = np.resize(img_with_idx, (h*w, c+2))
    img_train = img_with_idx[gt_train>0,:]
    img_test = img_with_idx[gt_test>0, :]
    img_whole = np.concatenate((img_train, img_test), axis=0)
    tr_gt = gt_train[gt_train>0].astype(int)
    te_gt = gt_test[gt_test>0].astype(int)
    whole_gt = np.concatenate((tr_gt, te_gt), axis=0)
    s2D_whole_spe, s2D_whole_spa = Mat_dis_s2(img_whole)
    H_whole_spe = construct_H_with_KNN_from_distance(s2D_whole_spe, k_spe, sig=1000)
    H_whole_spa = construct_H_with_KNN_from_distance(s2D_whole_spa, k_spa, sig=100)
    H_whole = np.concatenate((H_whole_spe, H_whole_spa), axis=1)
    whole_gt = convert_to_one_hot(whole_gt - 1, num_class)
    whole_gt = whole_gt.T
    a = tr_gt.shape[0]
    b = te_gt.shape[0]
    c = whole_gt.shape[0]
    mask_TR = sample_mask(np.arange(0, a), whole_gt.shape[0])
    mask_TE = sample_mask(np.arange(a, a + b), whole_gt.shape[0])
    if variable_weight == False:
        GHy_whole = _generate_G_from_H(H_whole_spe)
        #img_whole = img_whole[:,:-2]
        # G_whole = generate_Graph(H_whole_spe)
        return img_whole, whole_gt, GHy_whole, mask_TR, mask_TE
    if variable_weight == True:
        DV2_H, W, invDE_HT_DV2 = _generate_G_from_H(H_whole, variable_weight = variable_weight)
        #img_whole = img_whole[:, :-2]
        return img_whole, whole_gt, DV2_H, W, invDE_HT_DV2, mask_TR, mask_TE, img_idx, h, w






