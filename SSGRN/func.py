import torch
import numpy as np
import scipy.io as scio

import cv2

class load():
    # load dataset(indian_pines & pavia_univ.)
    def load_data(self, flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('../../Dataset/Indian_pines.mat')
            Ind_pines_gt_dict = scio.loadmat('../../Dataset/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            # remove the water absorption bands

            no_absorption = list(set(np.arange(0, 103)) | set(np.arange(108, 149)) | set(np.arange(163, 219)))

            original = Ind_pines_dict['indian_pines'][:, :, no_absorption].reshape(145 * 145, 200)

            print(original.shape)
            print('Remove wate absorption bands successfully!')

            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines'].shape[0]
            c = Ind_pines_dict['indian_pines'].shape[1]
            categories = 17
        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('../../Dataset/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('../../Dataset/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10

        if flag == 'houston':
            houst_dict = scio.loadmat('../../Dataset/Houston.mat')
            houst_gt_dict = scio.loadmat('../../Dataset/Houston_GT.mat')

            print(houst_dict['Houston'].shape)
            print(houst_gt_dict['Houston_GT'].shape)

            original = houst_dict['Houston'].reshape(349 * 1905, 144)
            gt = houst_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = houst_dict['Houston'].shape[0]
            c = houst_dict['Houston'].shape[1]
            categories = 16

        if flag == 'salina':
            salinas_dict = scio.loadmat('../../Dataset/Salinas_corrected.mat')
            salinas_gt_dict = scio.loadmat('../../Dataset/Salinas_gt.mat')

            print(salinas_dict['salinas_corrected'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas_corrected'].reshape(512 * 217, 204)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas_corrected'].shape[0]
            c = salinas_dict['salinas_corrected'].shape[1]
            categories = 17

        if flag == 'ksc':
            ksc_dict = scio.loadmat('../../Dataset/KSC.mat')
            ksc_gt_dict = scio.loadmat('../../Dataset/KSC_gt.mat')

            print(ksc_dict['KSC'].shape)
            print(ksc_gt_dict['KSC_gt'].shape)

            original = ksc_dict['KSC'].reshape(512 * 614, 176)
            original[original > 400] = 0
            gt = ksc_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = ksc_dict['KSC'].shape[0]
            c = ksc_dict['KSC'].shape[1]
            categories = 14

        rows = np.arange(gt.shape[0])  # start from 0
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt]

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled  data

        return All_data, labeled_data, rows_num, categories, r, c, flag


class product():
    def __init__(self, c, flag, All_data):
        self.c=c
        self.flag = flag
        self.All_data = All_data
    # product the training and testing pixel ID
    def generation_num(self, labeled_data, rows_num):

        train_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]  # all ID of a special class
            #print(i, temp_num.shape[0])
            #np.random.seed(2020)
            np.random.shuffle(temp_num)  # random sequence
            if self.flag == 'indian':
                if i == 1:
                    train_num.append(temp_num[0:33])
                elif i == 7:
                    train_num.append(temp_num[0:20])
                elif i == 9:
                    train_num.append(temp_num[0:14])
                elif i == 16:
                    train_num.append(temp_num[0:75])
                else:
                    train_num.append(temp_num[0:100])
            if self.flag == 'pavia' or self.flag=='houston' or self.flag=='salina':
                train_num.append(temp_num[0:100])
            if self.flag == 'ksc':
                if i==1:
                    train_num.append(temp_num[0:33])
                elif i==2:
                    train_num.append(temp_num[0:23])
                elif i==3:
                    train_num.append(temp_num[0:24])
                elif i==4:
                    train_num.append(temp_num[0:24])
                elif i==5:
                    train_num.append(temp_num[0:15])
                elif i==6:
                    train_num.append(temp_num[0:22])
                elif i==7:
                    train_num.append(temp_num[0:9])
                elif i==8:
                    train_num.append(temp_num[0:38])
                elif i==9:
                    train_num.append(temp_num[0:51])
                elif i==10:
                    train_num.append(temp_num[0:39])
                elif i==11:
                    train_num.append(temp_num[0:41])
                elif i==12:
                    train_num.append(temp_num[0:49])
                elif i==13:
                    train_num.append(temp_num[0:91])
        #             else:
        #                 train_num.append(temp_num[0:int(temp.shape[0]*0.1)])

        trn_num = [x for j in train_num for x in j]  # merge
        #np.random.seed(2020)
        np.random.shuffle(trn_num)
        val_num = trn_num[int(len(trn_num)*0.8):]
        tes_num = list(set(rows_num) - set(trn_num))
        pre_num = list(set(range(0, self.All_data.shape[0])) - set(trn_num))
        #trn_num = list(set(trn_num) | set(tes_num)) # for lichao mou's paper
        print('number of training sample', int(len(trn_num)*0.8))
        return rows_num, trn_num[:int(len(trn_num)*0.8)], val_num, tes_num, pre_num


    def production_label(self, num, y_map, split='Trn'):

        num = np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // self.c
        idx_2d[:, 1] = num % self.c

        label_map = np.zeros(y_map.shape)
        for i in range(num.shape[0]):
            label_map[idx_2d[i,0],idx_2d[i,1]] = self.All_data[num[i],-1]

        print('{} label map preparation Finished!'.format(split))
        return label_map

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]#output上分对的类别
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)#output上分对的类别中每类的个数
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)#output每类的个数
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)#target每类的个数
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()
