# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:22:09 2019

@author: 王天宁
"""

import numpy as np

def kappa(matrix):
    n=np.sum(matrix)
    sum_po=0
    sum_pe=0
    for i in range(len(matrix[0])):
        sum_po+=matrix[i][i]
        row=np.sum(matrix[i,:])
        col=np.sum(matrix[:,i])
        sum_pe+=row*col
    po=sum_po/n
    pe=sum_pe/(n*n)
    return (po-pe)/(1-pe)