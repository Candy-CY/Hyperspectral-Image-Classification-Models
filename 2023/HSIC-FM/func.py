import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class load():
    def load_data(self,flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('./Indian_pines_corrected.mat')
            Ind_pines_gt_dict = scio.loadmat('./Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines_corrected'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            original = Ind_pines_dict['indian_pines_corrected'].reshape(145 * 145, 200)
            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines_corrected'].shape[0]
            c = Ind_pines_dict['indian_pines_corrected'].shape[1]
            categories = 17
        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('./PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('./PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10
        if flag == 'ksc':
            ksc_dict = scio.loadmat('./KSC.mat')
            ksc_gt_dict=scio.loadmat('./KSC_gt.mat')

            print(ksc_dict['KSC'].shape)
            print(ksc_gt_dict['KSC_gt'].shape)

            original = ksc_dict['KSC'].reshape(512 * 614, 176)
            original[original>400]=0
            gt = ksc_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = ksc_dict['KSC'].shape[0]
            c = ksc_dict['KSC'].shape[1]
            categories = 14

        if flag == 'sali':
            salinas_dict = scio.loadmat('./Salinas_corrected.mat')
            salinas_gt_dict = scio.loadmat('./Salinas_gt.mat')

            print(salinas_dict['salinas_corrected'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas_corrected'].reshape(512 * 217, 204)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas_corrected'].shape[0]
            c = salinas_dict['salinas_corrected'].shape[1]
            categories = 17
        if flag == 'sali_a':
            salinas_a_dict = scio.loadmat('./SalinasA_corrected.mat')
            salinas_a_gt_dict = scio.loadmat('./SalinasA_gt.mat')

            print(salinas_a_dict['salinasA'].shape)
            print(salinas_a_gt_dict['salinasA_gt'].shape)

            original = salinas_a_dict['salinasA'].reshape(83 * 86, 204)
            gt = salinas_a_gt_dict['salinasA_gt'].reshape(83 * 86, 1)

            r = salinas_a_dict['salinasA'].shape[0]
            c = salinas_a_dict['salinasA'].shape[1]
            categories = 7
        if flag == 'hanchuan':
            hc_dict = scio.loadmat('./WHU_Hi_HanChuan.mat')
            hc_gt_dict=scio.loadmat('./WHU_Hi_HanChuan_gt.mat')

            print(hc_dict['WHU_Hi_HanChuan'].shape)
            print(hc_gt_dict['WHU_Hi_HanChuan_gt'].shape)

            original = hc_dict['WHU_Hi_HanChuan'].reshape(1217 * 303, 274)
            gt = hc_gt_dict['WHU_Hi_HanChuan_gt'].reshape(1217 * 303, 1)

            r = hc_dict['WHU_Hi_HanChuan'].shape[0]
            c = hc_dict['WHU_Hi_HanChuan'].shape[1]
            categories = 16+1
        if flag == 'houston':
            houston_dict = scio.loadmat('./Houston.mat')
            houston_gt_dict = scio.loadmat('./Houston_GT.mat')

            print(houston_dict['Houston'].shape)
            print(houston_gt_dict['Houston_GT'].shape)

            original = houston_dict['Houston'].reshape(349 * 1905, 144)
            gt = houston_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = houston_dict['Houston'].shape[0]
            c = houston_gt_dict['Houston_GT'].shape[1]
            categories = 15+1
        if flag == 'honghu':
            houston_dict = scio.loadmat('./WHU_Hi_HongHu.mat')
            houston_gt_dict = scio.loadmat('./WHU_Hi_HongHu_gt.mat')

            print(houston_dict['WHU_Hi_HongHu'].shape)
            print(houston_gt_dict['WHU_Hi_HongHu_gt'].shape)

            original = houston_dict['WHU_Hi_HongHu'].reshape(940 * 475, 270)
            gt = houston_gt_dict['WHU_Hi_HongHu_gt'].reshape(940 * 475, 1)

            r = houston_dict['WHU_Hi_HongHu'].shape[0]
            c = houston_gt_dict['WHU_Hi_HongHu_gt'].shape[1]
            categories = 22+1
        if flag == 'longkou':
            houston_dict = scio.loadmat('./WHU_Hi_LongKou.mat')
            houston_gt_dict = scio.loadmat('./WHU_Hi_LongKou_gt.mat')

            print(houston_dict['WHU_Hi_LongKou'].shape)
            print(houston_gt_dict['WHU_Hi_LongKou_gt'].shape)

            original = houston_dict['WHU_Hi_LongKou'].reshape(550 * 400, 270)
            gt = houston_gt_dict['WHU_Hi_LongKou_gt'].reshape(550 * 400, 1)

            r = houston_dict['WHU_Hi_LongKou'].shape[0]
            c = houston_gt_dict['WHU_Hi_LongKou_gt'].shape[1]
            categories = 9+1

        rows = np.arange(gt.shape[0])

        All_data = np.c_[rows, original, gt]


        if flag == 'hanchuan' or flag == 'honghu' or flag == 'longkou':
            All_data = All_data.astype(np.int64)


        # 剔除非0类别，获取所有labeled数据
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # 所有labeled数据的ID

        return All_data, labeled_data, rows_num, categories, r, c, flag

    ##无放回抽样
    def sampling(self,All_data,categories):
        K=10
        M=categories-1
        origin=All_data[:,1:-1]
        bands=np.arange(origin.shape[1])
        ensumb_num={}
        ensumb_feature_set={}
        for i in range(K):
            ensumb_num[str(i+1)]=np.random.choice(bands,M,replace=False)
            idx=[j for j in range(len(bands)) if bands[j] in ensumb_num[str(i+1)]]
            bands=np.delete(bands,idx)
        for i in range(K):
            ensumb_feature_set[str(i+1)]=origin[:,ensumb_num[str(i+1)]]
        return ensumb_feature_set,K,M

class product():
    def __init__(self,c,flag):
        self.c=c
        self.flag=flag
    def generation_num(self,labeled_data, rows_num, All_data):

        train_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]  # 某类别的所有ID
            np.random.shuffle(temp_num)  # 打乱顺序
            if self.flag == 'indian':
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

            if self.flag == 'pavia':
                train_num.append(temp_num[0:int(50)])
            if self.flag == 'ksc':
                train_num.append(temp_num[0:int(5)])
            if self.flag == 'houston':
                train_num.append(temp_num[0:int(50)])
            if self.flag == 'sali':
                train_num.append(temp_num[0:int(30)])
            if self.flag == 'hanchuan':
                train_num.append(temp_num[0:int(5)])
            if self.flag == 'PU_split':
                train_num.append(temp_num[0:int(10)])
            if self.flag == 'honghu':
                train_num.append(temp_num[0:int(5)])
            if self.flag == 'longkou':
                train_num.append(temp_num[0:int(5)])

        trn_num = [x for j in train_num for x in j]
        tes_num = list(set(rows_num) - set(trn_num))
        pre_num = list(set(range(0, All_data.shape[0])) - set(trn_num))
        return rows_num, trn_num, tes_num, pre_num


    def production_data_trn(self, rows_num, trn_num, half_s, image_3d_mat):
        trn_num = np.array(trn_num)
        idx_2d_trn = np.zeros([trn_num.shape[0], 2]).astype(int)
        idx_2d_trn[:, 0] = np.floor(trn_num / self.c)
        idx_2d_trn[:, 1] = trn_num + 1 - self.c * idx_2d_trn[:, 0] - 1
        # neibour area (2*half_s+1)
        patch_size=2*half_s+1
        trn_spat = np.zeros([trn_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        trn_spe = np.zeros([trn_num.shape[0], image_3d_mat.shape[2]])
        neibour_num = []
        for i in range(idx_2d_trn.shape[0]):
            # 图像扩展
            row = idx_2d_trn[i, 0] + half_s
            col = idx_2d_trn[i, 1] + half_s
            trn_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]
            # 把扩展后邻域像素ID映射回原图
            neibour_num = neibour_num + [(row + j - half_s) * self.c + col + k - half_s for j in range(-half_s, half_s+1) for k
                                         in range(-half_s, half_s+1)]
            trn_spe[i, :] = image_3d_mat[row, col, :]
        val_num = list(set(rows_num) - set(neibour_num))  # 差集(在rows_num不在 neibour_num): 防止数据偷看
        return trn_spe, trn_spat, trn_num, val_num

    def production_data_valtespre(self, tes_num, half_s, image_3d_mat, flag='Tes'):
        tes_num = np.array(tes_num)
        idx_2d_tes = np.zeros([tes_num.shape[0], 2]).astype(int)
        idx_2d_tes[:, 0] = np.floor(tes_num / self.c)
        idx_2d_tes[:, 1] = tes_num + 1 - self.c * idx_2d_tes[:, 0] - 1
        # neibour area(2*half_s+1)
        patch_size = 2 * half_s + 1
        tes_spat = np.zeros([tes_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        tes_spe = np.zeros([tes_num.shape[0], image_3d_mat.shape[2]])
        for i in range(idx_2d_tes.shape[0]):
            # 图像扩展
            row = idx_2d_tes[i, 0] + half_s
            col = idx_2d_tes[i, 1] + half_s
            tes_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]

            tes_spe[i, :] = image_3d_mat[row, col, :]
        return tes_spe, tes_spat,tes_num

    def production_data_trn_SpeAll(self, rows_num, trn_num, half_s, image_3d_mat, All_data_Spe_mat_origin):
        trn_num = np.array(trn_num)
        ##Training set(spatial)
        idx_2d_trn = np.zeros([trn_num.shape[0], 2]).astype(int)
        idx_2d_trn[:, 0] = np.floor(trn_num / self.c)
        idx_2d_trn[:, 1] = trn_num + 1 - self.c * idx_2d_trn[:, 0] - 1
        # neibour area(2*half_s+1)
        patch_size = 2 * half_s + 1
        trn_spat = np.zeros([trn_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        trn_spe = np.zeros([trn_num.shape[0], All_data_Spe_mat_origin.shape[2]])
        neibour_num = []
        for i in range(idx_2d_trn.shape[0]):
            # 图像扩展
            row = idx_2d_trn[i, 0] + half_s
            col = idx_2d_trn[i, 1] + half_s
            trn_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]
            # 把扩展后邻域像素ID映射回原图
            neibour_num = neibour_num + [(row + j - half_s) * self.c + col + k - half_s for j in
                                         range(-half_s, half_s + 1) for k
                                         in range(-half_s, half_s + 1)]
            trn_spe[i, :] = All_data_Spe_mat_origin[row, col, :]
        val_num = list(set(rows_num) - set(neibour_num))  # 差集(在rows_num不在neibour_num): 防止数据偷看

        return trn_spe, trn_spat, trn_num, val_num

    def production_data_valtespre_SpeAll(self, tes_num, half_s, image_3d_mat, All_data_Spe_mat_origin, flag='Tes'):
        tes_num = np.array(tes_num)
        idx_2d_tes = np.zeros([tes_num.shape[0], 2]).astype(int)
        idx_2d_tes[:, 0] = np.floor(tes_num / self.c)
        idx_2d_tes[:, 1] = tes_num + 1 - self.c * idx_2d_tes[:, 0] - 1
        # neibour area(2*half_s+1)
        patch_size = 2 * half_s + 1
        tes_spat = np.zeros([tes_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        tes_spe = np.zeros([tes_num.shape[0], All_data_Spe_mat_origin.shape[2]])
        for i in range(idx_2d_tes.shape[0]):
            # 图像扩展
            row = idx_2d_tes[i, 0] + half_s
            col = idx_2d_tes[i, 1] + half_s
            tes_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]
            tes_spe[i, :] = All_data_Spe_mat_origin[row, col, :]

        return tes_spe, tes_spat,tes_num

    def normlization(self, data_spat, mi, ma):

        scaler = MinMaxScaler(feature_range=(mi, ma))

        spat_data = data_spat.reshape(-1, data_spat.shape[-1])
        data_spat_new = scaler.fit_transform(spat_data).reshape(data_spat.shape)

        print('Dataset normalization Finished!')
        return data_spat_new


class preprocess():
    def __init__(self,t):
        self.transform=t
    def Dim_reduction(self, All_data):

        Alldata_DR=All_data

        if self.transform =='pca':
            pca_data_pre = All_data[:, 1:-1]  # except ID (0) and gt (-1)
            print(pca_data_pre.shape)
            pca_transformer = PCA(n_components=1)
            pca_data = pca_transformer.fit_transform(All_data[:, 1:-1])
            print(pca_data.shape)

            Alldata_DR = pca_data

            print('PCA Finished!')

        return Alldata_DR

def DrawResult(labels, imageID):
    labels -= 1
    num_class = labels.max() + 1
    if imageID == 1:  # PU
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
    elif imageID == 2:  # IP
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
    elif imageID == 5:  # Pavia Centre
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
    elif imageID == 6:  # KSC
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
    elif imageID == 7:  # Houston
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
    elif imageID == 9:  # Honghu
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
                            [238, 0, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [0, 139, 0],
                            [37, 58, 150],
                            [47, 78, 161],
                            [123, 18, 20]])
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

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    return X_result
