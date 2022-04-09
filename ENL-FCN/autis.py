import os
import torch
import numpy as np
import scipy.io as sio
import spectral as spy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from sklearn import metrics
import matplotlib.pyplot as plt

def accuracy(output, target, classcount):
#Computes the precision@k for the specified values of k"""
    output=output.view(classcount,-1)
    target=target.view(1,-1)
    #
    m,n=output.size()
    _,L_output=torch.topk(output, 1, 0, True)
    #print(torch.max(L_output))
    #print(torch.nonzero(L_output).size(0))
    count=0
    aa=0
    for i in range(n):
        #print(L_target.data[0,i])
        if target[0,i]!=0  and L_output[0,i]==target[0,i]:
            aa=aa+1
        if target[0,i]!=0:
            count=count+1

    return aa, count

def ClassificationAccuracy(output, target, classcount):
#Computes the precision@k for the specified values of k"""
    m, n = output.shape
    #output=np.reshape(output, [classcount, m*n])
    #target=np.reshape(target, [1, ])  #groundtruth label
    #m,n=output.size()
    #L_output=np.argmax(output, axis=0) # output label
    #_,L_output=torch.topk(output, 1, 0, True)

    correct_perclass=np.zeros([classcount-1])
    count_perclass = np.zeros([classcount-1])
    count=0
    aa=0

    for i in range(m):
        for j in range(n):
            if target[i, j]!=0:
                count=count+1
                count_perclass[int(target[i,j]-1)] += 1
                if output[i, j]==target[i, j]:
                    aa=aa+1
                    correct_perclass[int(target[i,j]-1)] += 1
            # if L_output[0,i]==7 or L_output[0,i]==9:
            #     print(target[0,i])
    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    test_OA=aa/count

    return test_AC_list, test_OA, test_AA, aa, count

# def ClassificationAccuracy(output, target, classcount):
# #Computes the precision@k for the specified values of k"""
#     output=output.view(classcount,-1)
#     target=target.view(1,-1)   #groundtruth label
#     m,n=output.size()
#     #_,L_output=np.argmax(output, axis=0) # output label
#     _,L_output=torch.topk(output, 1, 0, True)
#
#     correct_perclass=np.zeros([classcount-1])
#     count_perclass = np.zeros([classcount-1])
#     count=0
#     aa=0
#
#     for i in range(n):
#         if target[0,i]!=0:
#             count=count+1
#             count_perclass[int(target[0,i]-1)] += 1
#             if L_output[0,i]==target[0,i]:
#                 aa=aa+1
#                 correct_perclass[int(target[0,i]-1)] += 1
#             # if L_output[0,i]==7 or L_output[0,i]==9:
#             #     print(target[0,i])
#     test_AC_list = correct_perclass / count_perclass
#     test_AA = np.average(test_AC_list)
#     test_OA=aa/count
#
#     return test_AC_list, test_OA, test_AA, aa, count

def Kappa(output, target, classcount):
#Computes the precision@k for the specified values of k"""
    output=output
    target=target
    sizeOutput=np.shape(output)
    m=sizeOutput[0]
    n=sizeOutput[1]
    #output_data = np.transpose(output, (1,2,0))
    #idx = np.argmax(output, axis=0)

    test_pre_label_list = []
    test_real_label_list = []
    for ii in range(m):
        for jj in range(n):
            if target[ii][jj] != 0:
                test_pre_label_list.append(output[ii][jj])
                test_real_label_list.append(target[ii][jj])
    test_pre_label_list = np.array(test_pre_label_list)
    test_real_label_list = np.array(test_real_label_list)
    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))

    return kappa

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


def SpiltHSI(data, gt, split_size, edge):
    '''
    split HSI data with given slice_number
    :param data: 3D HSI data
    :param gt: 2D ground truth
    :param split_size: [height_slice,width_slice]
    :return: splited data and corresponding gt
    '''
    e = edge  # 补边像素个数
    #height, width, band=np.shape(data)
    #def GT_To_One_Hot(gt, class_count):
        # GT_One_Hot = []  # 转化为one-hot形式的标签
        # for i in range(gt.shape[0]):
        #     for j in range(gt.shape[1]):
        #         temp = np.zeros(class_count, dtype=np.float32)
        #         if gt[i, j] != 0:
        #             temp[int(gt[i, j]) - 1] = 1
        #         GT_One_Hot.append(temp)
        # GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        # return GT_One_Hot

    split_height = split_size[0]
    split_width = split_size[1]
    m, n, d = data.shape
    #gt = np.reshape(gt, [m, n])
    GT=gt
    #GT = GT_To_One_Hot(gt, class_count)

    # 将无法整除的块补0变为可整除
    if m % split_height != 0 or n % split_width != 0:
        data = np.pad(data, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                      mode='constant')
        GT = np.pad(GT, [[0, split_height - m % split_height], [0, split_width - n % split_width]],
                    mode='constant')
    m_height = int(data.shape[0] / split_height)
    m_width = int(data.shape[1] / split_width)

    pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
    pad_GT = np.pad(GT, [[e, e], [e, e]], mode="constant")
    final_data = []
    final_gt=[]
    for i in range(split_height):
        for j in range(split_width):
            temp1 = pad_data[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e, :]
            temp2 = pad_GT[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e]
            final_data.append(temp1)
            final_gt.append(temp2)
    # gt_split = np.split(GT, split_height, 0)
    # final_gt = []
    # for i in range(gt_split.__len__()):
    #     temp_gt = np.split(gt_split[i], split_width, 1)
    #     for j in range(temp_gt.__len__()):
    #         tt = temp_gt[j]
    #         tt = np.pad(tt, [[e, e], [e, e], [0, 0]], mode="constant")
    #         final_gt.append(tt)
    final_data = np.array(final_data)
    final_gt = np.array(final_gt)

    return final_data, final_gt

def PatchStack(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):

    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:,
                                                                                                               :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE, :]

    HSI_stack = np.argmax(HSI_stack, axis=2)
    HSI_stack = HSI_stack[0: -(split_height - m % split_height), 0: -(split_width - n % split_width)]

    return HSI_stack

