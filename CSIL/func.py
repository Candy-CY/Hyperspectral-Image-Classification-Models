import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


class load():
    def load_data(self, flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('./HSI_datasets/Indian_pines_corrected.mat')
            Ind_pines_gt_dict = scio.loadmat('/HSI_datasets/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines_corrected'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            original = Ind_pines_dict['indian_pines_corrected'].reshape(145 * 145, 200)
            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines_corrected'].shape[0]
            c = Ind_pines_dict['indian_pines_corrected'].shape[1]
            categories = 17
        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('/HSI_datasets/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('/HSI_datasets/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10
        if flag == 'ksc':
            ksc_dict = scio.loadmat('/HSI_datasets/KSC.mat')
            ksc_gt_dict = scio.loadmat('/HSI_datasets/KSC_gt.mat')

            print(ksc_dict['KSC'].shape)
            print(ksc_gt_dict['KSC_gt'].shape)

            original = ksc_dict['KSC'].reshape(512 * 614, 176)
            original[original > 400] = 0
            gt = ksc_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = ksc_dict['KSC'].shape[0]
            c = ksc_dict['KSC'].shape[1]
            categories = 14
        if flag == 'sali':
            salinas_dict = scio.loadmat('/HSI_datasets/Salinas_corrected.mat')
            salinas_gt_dict = scio.loadmat('/HSI_datasets/Salinas_gt.mat')

            print(salinas_dict['salinas_corrected'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas_corrected'].reshape(512 * 217, 204)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas_corrected'].shape[0]
            c = salinas_dict['salinas_corrected'].shape[1]
            categories = 17
        if flag == 'sali_a':
            salinas_a_dict = scio.loadmat('/HSI_datasets/SalinasA_corrected.mat')
            salinas_a_gt_dict = scio.loadmat('/HSI_datasets/SalinasA_gt.mat')

            print(salinas_a_dict['salinasA'].shape)
            print(salinas_a_gt_dict['salinasA_gt'].shape)

            original = salinas_a_dict['salinasA'].reshape(83 * 86, 224)
            gt = salinas_a_gt_dict['salinasA_gt'].reshape(83 * 86, 1)

            r = salinas_a_dict['salinasA'].shape[0]
            c = salinas_a_dict['salinasA'].shape[1]
            categories = 7
        if flag == 'houston':
            ksc_dict = scio.loadmat('/HSI_datasets/Houston.mat')
            ksc_gt_dict = scio.loadmat('/HSI_datasets/Houston_GT.mat')

            print(ksc_dict['Houston'].shape)
            print(ksc_gt_dict['Houston_GT'].shape)

            original = ksc_dict['Houston'].reshape(349 * 1905, 144)
            gt = ksc_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = ksc_dict['Houston'].shape[0]
            c = ksc_dict['Houston'].shape[1]
            categories = 16
        if flag == 'hanchuan':
            hc_dict = scio.loadmat('/HSI_datasets/WHU_Hi_HanChuan.mat')
            hc_gt_dict = scio.loadmat('/HSI_datasets/WHU_Hi_HanChuan_gt.mat')

            print(hc_dict['WHU_Hi_HanChuan'].shape)
            print(hc_gt_dict['WHU_Hi_HanChuan_gt'].shape)

            original = hc_dict['WHU_Hi_HanChuan'].reshape(1217 * 303, 274)
            gt = hc_gt_dict['WHU_Hi_HanChuan_gt'].reshape(1217 * 303, 1)

            r = hc_dict['WHU_Hi_HanChuan'].shape[0]
            c = hc_dict['WHU_Hi_HanChuan'].shape[1]
            categories = 16 + 1
        if flag == 'honghu':
            houston_dict = scio.loadmat('/HSI_datasets/WHU_Hi_HongHu.mat')
            houston_gt_dict = scio.loadmat('/HSI_datasets/WHU_Hi_HongHu_gt.mat')

            print(houston_dict['WHU_Hi_HongHu'].shape)
            print(houston_gt_dict['WHU_Hi_HongHu_gt'].shape)

            original = houston_dict['WHU_Hi_HongHu'].reshape(940 * 475, 270)
            gt = houston_gt_dict['WHU_Hi_HongHu_gt'].reshape(940 * 475, 1)

            r = houston_dict['WHU_Hi_HongHu'].shape[0]
            c = houston_gt_dict['WHU_Hi_HongHu_gt'].shape[1]
            categories = 22 + 1
        if flag == 'longkou':
            hc_dict = scio.loadmat('/HSI_datasets/WHU_Hi_LongKou.mat')
            hc_gt_dict = scio.loadmat('/HSI_datasets/WHU_Hi_LongKou_gt.mat')

            print(hc_dict['WHU_Hi_LongKou'].shape)
            print(hc_gt_dict['WHU_Hi_LongKou_gt'].shape)

            original = hc_dict['WHU_Hi_LongKou'].reshape(550 * 400, 270)
            gt = hc_gt_dict['WHU_Hi_LongKou_gt'].reshape(550 * 400, 1)

            r = hc_dict['WHU_Hi_LongKou'].shape[0]
            c = hc_dict['WHU_Hi_LongKou'].shape[1]
            categories = 9 + 1
        if flag == 'HU2018':
            hc_dict = scio.loadmat('./Dataset/HSI/Houston2018.mat')
            hc_gt_dict=scio.loadmat('./Dataset/HSI/Houston2018_GT.mat')

            print(hc_dict['Houston2018'].shape)
            print(hc_gt_dict['Houston2018_GT'].shape)

            original = hc_dict['Houston2018'].reshape(601 * 2384, 48)
            gt = hc_gt_dict['Houston2018_GT'].reshape(601 * 2384, 1)

            r = hc_dict['Houston2018'].shape[0]
            c = hc_dict['Houston2018'].shape[1]
            categories = 20+1
        if flag == 'paviaC':
            pav_univ_dict = scio.loadmat('/HSI_datasets/Pavia.mat')
            pav_univ_gt_dict = scio.loadmat('/HSI_datasets/Pavia_gt.mat')

            print(pav_univ_dict['pavia'].shape)
            print(pav_univ_gt_dict['pavia_gt'].shape)

            original = pav_univ_dict['pavia'].reshape(1096 * 715, 102)
            gt = pav_univ_gt_dict['pavia_gt'].reshape(1096 * 715, 1)

            r = pav_univ_dict['pavia'].shape[0]
            c = pav_univ_dict['pavia'].shape[1]
            categories = 9 + 1


        rows = np.arange(gt.shape[0])  # 从0开始
        # 行号(ID)，特征数据，类别号
        All_data = np.c_[rows, original, gt]
        ######################
        if flag == 'hanchuan' or flag == 'honghu' or flag == 'longkou':  # 'All_data' of hanchuan dataset is float, which needs to be changed into int
            All_data = All_data.astype(np.int64)
        ######################
        # 剔除非0类别，获取所有labeled数据
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # 所有labeled数据的ID

        return All_data, labeled_data, rows_num, categories, r, c, flag

    ##无放回抽样(selected without replacement)
    def sampling(self, All_data, categories):
        K = 10  ##ICA的分组
        M = categories - 1  # 每组多少个
        origin = All_data[:, 1:-1]
        bands = np.arange(origin.shape[1])
        ensumb_num = {}  # 波段号
        ensumb_feature_set = {}
        for i in range(K):
            ensumb_num[str(i + 1)] = np.random.choice(bands, M, replace=False)
            idx = [j for j in range(len(bands)) if bands[j] in ensumb_num[str(i + 1)]]
            bands = np.delete(bands, idx)  # 删除idx处的值
        for i in range(K):
            ensumb_feature_set[str(i + 1)] = origin[:, ensumb_num[str(i + 1)]]
        return ensumb_feature_set, K, M


class product():
    def __init__(self, c, flag):
        self.c = c
        self.flag = flag

    # training and testing pixel ID
    def generation_num(self, labeled_data, rows_num, All_data):

        train_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]  # 某类别的所有ID
            np.random.shuffle(temp_num)  # 打乱顺序
            if self.flag == 'indian':
                # Trn-10:  Sum 1105
                if i == 1:
                    train_num.append(temp_num[0:20])
                elif i == 7:
                    train_num.append(temp_num[0:10])
                elif i == 9:
                    train_num.append(temp_num[0:5])
                elif i == 15:
                    train_num.append(temp_num[0:40])
                elif i == 16:
                    train_num.append(temp_num[0:40])
                else:
                    train_num.append(temp_num[0:90])

            if self.flag == 'pavia' or self.flag == 'Berlin' or self.flag == 'paviaC':
                train_num.append(temp_num[0:int(50)])
            if self.flag == 'ksc':
                train_num.append(temp_num[0:int(5)])

            if self.flag == 'houston':
                train_num.append(temp_num[0:int(50)])
            if self.flag == 'sali':
                train_num.append(temp_num[0:int(50)])
            if self.flag == 'hanchuan':
                train_num.append(temp_num[0:int(30)])
            if self.flag == 'honghu':
                train_num.append(temp_num[0:int(5)])
            if self.flag == 'longkou':
                train_num.append(temp_num[0:int(5)])
            if self.flag == 'HU2018':
                train_num.append(temp_num[0:int(100)])

        trn_num = [x for j in train_num for x in j]  # 合并list中各元素
        tes_num = list(set(rows_num) - set(trn_num))  # ID_labeled - ID_trn = ID_tes
        pre_num = list(set(range(0, All_data.shape[0])) - set(trn_num) - set(tes_num))
        print('number of training sample', len(trn_num))
        print('number of test sample', len(tes_num))
        return rows_num, trn_num, tes_num, pre_num

    def production_data_trn(self, rows_num, trn_num, half_L, image_3d_mat_L, half_S, image_3d_mat_S):
        trn_num = np.array(trn_num)
        idx_2d_trn = np.zeros([trn_num.shape[0], 2]).astype(int)
        idx_2d_trn[:, 0] = np.floor(trn_num / self.c)
        idx_2d_trn[:, 1] = trn_num + 1 - self.c * idx_2d_trn[:, 0] - 1
        patch_size_L = 2 * half_L + 1
        patch_size_S = 2 * half_S + 1
        trn_spat_L = np.zeros([trn_num.shape[0], patch_size_L, patch_size_L, image_3d_mat_L.shape[2]], 'float32')
        trn_spat_S = np.zeros([trn_num.shape[0], patch_size_S, patch_size_S, image_3d_mat_S.shape[2]], 'float32')
        neighbour_num = []
        for i in range(idx_2d_trn.shape[0]):
            row = idx_2d_trn[i, 0] + half_L  # row num of center pixel in Mirror patch
            col = idx_2d_trn[i, 1] + half_L  # col num of center pixel in Mirror patch
            trn_spat_L[i, :, :, :] = image_3d_mat_L[(row - half_L):row + half_L + 1,
                                     (col - half_L):col + half_L + 1, :]
            # 把扩展后邻域像素ID映射回原图
            neighbour_num = neighbour_num + [(row + j - half_L) * self.c + col + k - half_L for j in
                                         range(-half_L, half_L + 1) for k
                                         in range(-half_L, half_L + 1)]

            row = idx_2d_trn[i, 0] + half_S
            col = idx_2d_trn[i, 1] + half_S
            trn_spat_S[i, :, :, :] = image_3d_mat_S[(row - half_S):row + half_S + 1,
                                     (col - half_S):col + half_S + 1, :]
            # 把扩展后邻域像素ID映射回原图
            neighbour_num = neighbour_num + [(row + j - half_S) * self.c + col + k - half_S for j in
                                         range(-half_S, half_S + 1) for k
                                         in range(-half_S, half_S + 1)]

        val_num = list(set(rows_num) - set(neighbour_num))  # 差集(在rows_num不在neibour_num): 防止数据偷看

        print('Training Spatial dataset preparation Finished!')
        return trn_spat_L, trn_spat_S, trn_num, val_num

    def production_data_valtespre(self, tes_num, half_L, image_3d_mat_L, half_S, image_3d_mat_S, flag='Tes'):
        tes_num = np.array(tes_num)
        idx_2d_tes = np.zeros([tes_num.shape[0], 2]).astype(int)
        idx_2d_tes[:, 0] = np.floor(tes_num / self.c)
        idx_2d_tes[:, 1] = tes_num + 1 - self.c * idx_2d_tes[:, 0] - 1

        patch_size_L = 2 * half_L + 1
        tes_spat_L = np.zeros([tes_num.shape[0], patch_size_L, patch_size_L, image_3d_mat_L.shape[2]], 'float32')
        for i in range(idx_2d_tes.shape[0]):
            # 图像扩展
            row = idx_2d_tes[i, 0] + half_L
            col = idx_2d_tes[i, 1] + half_L
            tes_spat_L[i, :, :, :] = image_3d_mat_L[(row - half_L):row + half_L + 1,
                                     (col - half_L):col + half_L + 1, :]

        patch_size_S = 2 * half_S + 1
        tes_spat_S = np.zeros([tes_num.shape[0], patch_size_S, patch_size_S, image_3d_mat_S.shape[2]], 'float32')
        for i in range(idx_2d_tes.shape[0]):
            # 图像扩展
            row = idx_2d_tes[i, 0] + half_S
            col = idx_2d_tes[i, 1] + half_S
            tes_spat_S[i, :, :, :] = image_3d_mat_S[(row - half_S):row + half_S + 1,
                                     (col - half_S):col + half_S + 1, :]
        return tes_spat_L, tes_spat_S, tes_num

    def normlization(self, data_spat, mi, ma):
        spat_data = data_spat.reshape(-1, data_spat.shape[-1])

        scaler = MinMaxScaler(feature_range=(mi, ma))
        data_spat_new = scaler.fit_transform(spat_data).reshape(data_spat.shape)

        print('Dataset normalization Finished!')
        return data_spat_new


class preprocess():
    def __init__(self, t):
        self.transform = t

    def Dim_reduction(self, All_data, numPCA):
        pca_time1 = time.time()
        Alldata_DR = All_data
        print('\nAll_data: ', All_data.shape)

        # if self.transform =='ica':
        #     ica_data_pre = All_data[:, 1:-1]
        #     print(ica_data_pre.shape)
        #     transformer = FastICA(n_components=50, whiten=True, random_state=None)
        #     fastica_data = transformer.fit_transform(ica_data_pre)
        #     print(fastica_data.shape)
        #
        #     Alldata_DR = fastica_data
        #
        #     print('ICA Finished!')

        if self.transform == 'pca':
            pca_data_pre = All_data[:, 1:-1]
            print(pca_data_pre.shape)
            pca_transformer = PCA(n_components=numPCA)  # n_components = Vit里参数channels
            pca_data = pca_transformer.fit_transform(All_data[:, 1:-1])
            print(pca_data.shape)

            Alldata_DR = pca_data

            pca_time2 = time.time()
            pca_time = pca_time2 - pca_time1
            print('PCA Finished!')
            print('pca_time: ', pca_time)

        return Alldata_DR


def DrawResult(labels, imageID):
    labels -= 1
    num_class = labels.max() + 1
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
    elif imageID == 3:  # Botswana
        row = 1476
        col = 256
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
                            [127, 255, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 4:  # Salinas
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
    elif imageID == 5:  # Pavia Center
        row = 1096
        col = 715
        palette = np.array([[37, 97, 163],
                            [44, 153, 60],
                            [122, 182, 41],
                            [219, 36, 22],
                            [227, 156, 47],
                            [227, 221, 223],
                            [108, 35, 127],
                            [130, 67, 142],
                            [229, 225, 74]])
        palette = palette * 1.0 / 255
    elif imageID == 6:
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 7:
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
    elif imageID == 8:  # Hanchuan
        row = 1217
        col = 303
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
    if imageID == 9:  # longkou
        row = 550
        col = 400
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
    elif imageID == 10:  # Longkou
        row = 550
        col = 400
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255
    elif imageID == 11 or imageID == 12:  # HU2018; Berlin
        row = 601
        col = 2384
        palette = np.array([[0, 255, 0],
                            [129, 255, 0],
                            [47, 140, 88],
                            [0, 139, 0],
                            [0, 70, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 192, 216],
                            [255, 0, 0],
                            [170, 160, 150],
                            [128, 128, 128],
                            [160, 0, 0],
                            [81, 1, 3],
                            [233, 162, 25],
                            [255, 255, 0],
                            [238, 154, 0],
                            [255, 0, 255],
                            [0, 0, 255],
                            [176, 196, 222]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    # plt.axis("off")
    # plt.imshow(X_result)
    return X_result
