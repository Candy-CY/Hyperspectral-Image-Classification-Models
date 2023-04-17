import numpy as np
# import matplotlib.pyplot as plt
from operator import truediv
import csv,threading
from sklearn import manifold
import torch.nn.functional as F
import torch
import scipy
# import cv2
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):#model的第一个Sequential()是有多层，所以遍历
            if index==5:
                continue
            x = layer(x)#torch.Size([1, 64, 55, 55])生成了64个通道
            if index==4:
                x = torch.squeeze(x, 2)
            if k == index:
                return x
def show_feature_map(feature_map, feature_data, i):
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])
#   现在4维，包括一个batch，测试的时候可以把batch去掉
    # 以下4行，通过双线性插值的方式改变保存图像的大小
    unsample = torch.nn.UpsamplingBilinear2d(size=(64, 64))
    feature_data = torch.sum(feature_data, 2)
    # feature_data = feature_data.view(1, feature_data.shape[2],
    #                                feature_data.shape[3], feature_data.shape[4])  # (1,64,55,55)
    # feature_data = torch.sum(feature_data, 1)
    feature_data = unsample(feature_data)
    feature_data = np.array(feature_data.cpu())
    # upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    # feature_map = upsample(feature_map)
    # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2],
    #                                feature_map.shape[3])
    feature_data = feature_data.squeeze(0)
    feature_data = feature_data.squeeze(0)
    feature_map = torch.sum(feature_map, 0)
    feature_map = feature_map.view(1, 1,feature_map.shape[0], feature_map.shape[1])  # (1,64,55,55)
    feature_map = unsample(feature_map)
    feature_map = np.array(feature_map.cpu())
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.squeeze(0)
    img_add = cv2.addWeighted(feature_data, 0.7, feature_map, 0.3, 0)
    # feature_map_num = feature_map.shape[0]  # 返回通道数
    # row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    plt.imshow(img_add, cmap='jet')  # feature_map[0].shape=torch.Size([55, 55])
    plt.axis('off')
    plt.show()
    plt.imsave('./feature_map_save/'+ str(i)+'.svg', img_add)
    plt.imsave('./feature_map_save/'+ str(i)+'_origin.svg', feature_data)
    # for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
    #     plt.subplot(row_num, row_num, index)
    #     plt.imshow(feature_map[index - 1].cpu(), cmap='jet')  # feature_map[0].shape=torch.Size([55, 55])
    #     # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
    #     plt.axis('off')
    #     # scipy.imsave('./feature_map_save/' + str(index) + ".png", feature_map[index - 1].cpu())
    # plt.show()
def DrawCluster(label, cluster, oa):

    label = np.array(label)
    palette = np.array([[0, 139, 0],
                        [0, 0, 255],
                        [255, 255, 0],
                        [255, 127, 80],
                        [255, 0, 255],
                        [139, 139, 0],
                        [0, 139, 139],
                        [0, 255, 0],
                        [0, 255, 255],
                        [0, 30, 190],
                        [127, 255, 0],
                        [218, 112, 214],
                        [46, 139, 87],
                        [0, 0, 139],
                        [255, 165, 0],
                        [127, 255, 212],
                        [218, 112, 214],
                        [255, 0, 0],
                        [205, 0, 0],
                        [139, 0, 0],
                        [65, 105, 225],
                        [240, 230, 140],
                        [244, 164, 96]])
    # palette = np.array([[0, 0, 255],
    #                     [76, 230, 0],
    #                     [255, 190, 232],
    #                     [255, 0, 0],
    #                     [156, 156, 156],
    #                     [255, 255, 115],
    #                     [0, 255, 197],
    #                     [132, 0, 168],
    #                     [0, 0, 0]])
    palette = palette * 1.0 / 255
    tsne = manifold.TSNE(n_components=2,init='pca')
    X_tsne = tsne.fit_transform(cluster)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)
    plt.figure()

    for i in range(23):
        xx1 = X_norm[np.where(label==i), 0]
        yy1 = X_norm[np.where(label==i), 1]

        plt.scatter(xx1,yy1, color=palette[i].reshape(1,-1),s=20, linewidths=2)
    plt.xlim(np.min(X_norm)-0.0001, np.max(X_norm)+0.0001)
    plt.ylim(np.min(X_norm)-0.0001, np.max(X_norm)+0.0001)
    # plt.legend(['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
    #                     'Fallow_smooth',
    #                     'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
    #                     'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
    #                     'Vinyard_untrained', 'Vinyard_vertical_trellis'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig('./CLUSTER/' + 'HHK'+ str('.%6f'%oa)+'.svg',dpi=600, bbox_inches='tight')
    plt.show()
    # plt.savefig("./cluster.png")
    # Y = tsne
def DrawResult(y_pred, imageID, OA):

    # ID=1:Pavia University
    # ID=2:Indian Pines
    # ID=6:KSC
    # labels = labels + 1
    num_class = y_pred.max()
    if imageID == 'PU':
        row = 610
        col = 340
        palette = np.array([[0, 0, 255],
                            [76, 230, 0],
                            [255, 190, 232],
                            [255, 0, 0],
                            [156, 156, 156],
                            [255, 255, 115],
                            [0, 255, 197],
                            [132, 0, 168],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 'IP':
        print('?')
        row = 145
        col = 145
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 'SA':
        row = 512
        col = 217
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 'PD':
        row = 377
        col = 512
        palette = np.array([[237, 227, 81],
                            [167, 237, 81],
                            [0, 0, 0],
                            [181, 117, 14],
                            [77, 122, 15],
                            [186, 186, 186]])
        palette = palette * 1.0 / 255
    elif imageID == 'HHK':
        row = 1147
        col = 1600
        palette = np.array([[0, 139, 0],
                  [0, 0, 255],
                  [255, 255, 0],
                  [255, 127, 80],
                  [255, 0, 255],
                  [139, 139, 0],
                  [0, 139, 139],
                  [0, 255, 0],
                  [0, 255, 255],
                  [0, 30, 190],
                  [127, 255, 0],
                  [218, 112, 214],
                  [46, 139, 87],
                  [0, 0, 139],
                  [255, 165, 0],
                  [127, 255, 212],
                  [218, 112, 214],
                  [255, 0, 0],
                  [205, 0, 0],
                  [139, 0, 0],
                  [65, 105, 225],
                  [240, 230, 140],
                  [244, 164, 96]])
        palette = palette * 1.0 / 255
    X_result = np.zeros((y_pred.shape[0],3))
    for i in range(0, num_class+1):
        X_result[np.where(y_pred == i), 0] = palette[i, 0]
        X_result[np.where(y_pred == i), 1] = palette[i, 1]
        X_result[np.where(y_pred == i), 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))

    # X_mask[1:-1,1:-1,:] = X_result
    plt.axis("off")
    plt.imsave('./image_result/'+imageID + '_s3net' + str(OA) + '.svg',X_result)
    return X_result
def neibor_result_choose(y_pred):
    y_pred_count = []
    new_y_pred = []
    for i in range(len(y_pred)):
        y_pred_count.append(y_pred[i])
        if (i+1) % 25 == 0:
            counts = np.bincount(y_pred_count)
            index = np.argmax(counts)
            max_time = np.max(counts)
            counts[index] = 0
            if max_time in counts:
                new_y_pred.append(np.argmax(y_pred_count[4]))
            else:
                new_y_pred.append(index)
            y_pred_count.clear()
            y_pred_count = []
    return new_y_pred
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def information_process(dataset, windowSize1, windowSize2, perclass, batch_size, iteration, K, add_info, Each_class_acc, margin):
    res = []
    for i in range(len(np.mean(Each_class_acc, 0))):
        str_ = str('%.2f' % np.mean(Each_class_acc, 0)[i]) + '+-' + str('%.2f' % np.std(Each_class_acc, 0)[i])
        res.append(str_)
    infomation = [dataset, 'windowSize1:',windowSize1,'windowSize2',windowSize2, 'perclass:', perclass, 'margin:', margin, 'iteration:', iteration, 'PCA:',K, 'batch_size:',batch_size, 'oa:',np.mean(add_info, 0)[0],
                  '+-', np.std(add_info, 0)[0], 'kappa:', np.mean(add_info, 0)[1], '+-', np.std(add_info, 0)[1], 'aa:',np.mean(add_info, 0)[2], '+-', np.std(add_info, 0)[2]
                  ,'each_acc:', res]
    print('oa:', np.mean(add_info, 0)[0], '+-', np.std(add_info, 0)[0], 'kappa:', np.mean(add_info, 0)[1], '+-',
          np.std(add_info, 0)[1], 'aa:', np.mean(add_info, 0)[2], '+-', np.std(add_info, 0)[2])
    csvFile = open("./Final_Experiment.csv", "a")
    writer = csv.writer(csvFile)
    writer.writerow(infomation)
    csvFile.close()

