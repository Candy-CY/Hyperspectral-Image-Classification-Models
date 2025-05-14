# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:57:44 2022

@author: HLB
"""
import numpy as np
import sklearn

import matplotlib.pyplot as plt

indianpines_colors = np.array([[0, 0, 0],
                                [255, 0, 0], [0,  255,  0], [0, 0,    255], [255,   255, 0],
                                [0,   255, 255], [255,  0,  255], [192,   192, 192], [128,  128,   128],
                                [128, 0,  0], [128, 128, 0], [0, 128, 0], [128,   0, 128],
                                [0, 128,  128], [0, 0,  128], [255, 165, 0], [255, 215,   0]])
indianpines_colors = sklearn.preprocessing.minmax_scale(indianpines_colors, feature_range=(0, 1))

def classification_map(img, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(img)
    fig.savefig(save_path, dpi=dpi)
    
    return 0 


def generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, OA, halfsize, dataset, day_str, num, model_name):  
    number_of_rows = np.size(image,0)
    number_of_columns = np.size(image,1)
     
    gt_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    predicted_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            gt_thematic_map[i, j, :] = indianpines_colors[gt[i,j]]
            predicted_thematic_map[i, j, :] = indianpines_colors[gt[i,j]]
    nclass = np.max(gt)
    
    fl = 0
    for i in range(nclass):
        print('test lable of class:',i)
        matrix = index[i]
        temprow = matrix[:,0]
        tempcol = matrix[:,1]
        m = len(temprow)
        fl = fl - nTrain_perClass[i] - nvalid_perClass[i]
        for j in range(nTrain_perClass[i] + nvalid_perClass[i], m):
            predicted_thematic_map[temprow[j], tempcol[j], :] = indianpines_colors[test_pred[fl + j]+1]
        fl = fl + m
        
 
    predicted_thematic_map = predicted_thematic_map[halfsize:number_of_rows -halfsize,halfsize:number_of_columns-halfsize,: ]
    gt_thematic_map = gt_thematic_map[halfsize:number_of_rows -halfsize,halfsize:number_of_columns-halfsize,: ]     
    path = '.'
    classification_map(predicted_thematic_map, gt, 600,
                        path + '/classification_maps/' + dataset + '_' + day_str +'_' + str(num) +'_OA_'+ str(round(OA, 2)) + '_' + model_name +  '.png')
    
    return predicted_thematic_map, gt_thematic_map


