# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                   [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                   [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
                   [0, 0, 128]])

def imgDraw(label, imgName, path='./pictures', show=True):
    """
    功能：根据标签绘制RGB图
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，label有效范围[1,num]
    """
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  # 对背景设置为黑色
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return Y_RGB


