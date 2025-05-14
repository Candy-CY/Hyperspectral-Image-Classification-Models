#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class Dataset
    Generate simulation data
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy
import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
import xlrd
import cv2

class Dataset(data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args
        self.sp_matrix = self.get_spectral_response(self.args.data_name)
        self.sp_range = self.get_sp_range(self.sp_matrix)
        self.msi_channel = self.sp_matrix.shape[1]
        self.hsi_channel = self.sp_matrix.shape[0]
        self.PSF = self.matlab_style_gauss2D(shape=(self.args.scale_factor,self.args.scale_factor),sigma=self.args.sigma)
        data_folder = os.path.join(self.args.default_datapath, args.data_name)
        if os.path.exists(data_folder):
            data_path = os.path.join(data_folder, "*.mat")
        else:
            return 0

        self.imgpath_list = sorted(glob.glob(data_path))
        self.img_list = []
        for i in range(len(self.imgpath_list)):
            self.img_list.append(io.loadmat(self.imgpath_list[i])['REF'])
        (_, _, self.hsi_channels) = self.img_list[0].shape

        temp_array = np.empty([args.image_size,args.image_size,self.hsi_channels], dtype=numpy.float64)

        for j in range(0,self.hsi_channels):
            temp_array[:,:,j] = cv2.flip(self.img_list[0][:,:,j],1)
        self.img_list.append(temp_array)

        '''generate simulation data'''
        self.img_patch_list = []
        self.img_lr_list = []
        self.img_msi_list = []
        self.img_lrmsi_frommsi_list= []
        self.img_lrmsi_fromlrhsi_list= []
        
        for i, img in enumerate(self.img_list):
            print(i)
            (h, w, c) = img.shape
            print(img.shape)
            s = self.args.scale_factor
            print(c)
            """Ensure that the side length can be divisible"""
            r_h, r_w = h%s, w%s
            img_patch = img[int(r_h/2):h-(r_h-int(r_h/2)),int(r_w/2):w-(r_w-int(r_w/2)),:]

            self.img_patch_list.append(img_patch)

            """low HSI"""
            img_lr = self.generate_low_HSI(img_patch, s)
            self.img_lr_list.append(img_lr)
            
            """high MSI"""
            img_msi = self.generate_MSI(img_patch, self.sp_matrix)
            print(img_msi.shape)
            self.img_msi_list.append(img_msi)
            
            """low MSI1 generated from high MSI"""
            lrmsi_1 = self.generate_low_HSI(img_msi, s)
            self.img_lrmsi_frommsi_list.append(lrmsi_1)
            
            """low MSI2 generated from low HSI"""
            lrmsi_2= self.generate_MSI(img_lr, self.sp_matrix)
            self.img_lrmsi_fromlrhsi_list.append(lrmsi_2)

    def matlab_style_gauss2D(self,shape=(3,3),sigma=2): 
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h   
    
    # obtain the sepctral response stored in  data//spectral_response//houston18.xls
    def get_spectral_response(self,data_name):
        xls_path = os.path.join(self.args.sp_root_path, data_name + '.xls')
        if not os.path.exists(xls_path):
            raise Exception("spectral response path does not exist")
        data = xlrd.open_workbook(xls_path)
        table = data.sheets()[0]
        num_cols = table.ncols
        cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(0,num_cols)]
        sp_data = np.concatenate(cols_list, axis=1)
        sp_data = sp_data / (sp_data.sum(axis=0))  #normalize the sepctral response
        return sp_data   
    
    #obtain the coverage index between multispectral spectral response and hyperspectral wavelength
    def get_sp_range(self,sp_matrix):
        
        HSI_bands, MSI_bands = sp_matrix.shape
        assert(HSI_bands>MSI_bands)
        sp_range = np.zeros([MSI_bands,2])
        for i in range(0,MSI_bands):
            index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
            sp_range[i,0] = index_dim_0[0] 
            sp_range[i,1] = index_dim_0[-1]
        return sp_range
    
    def downsamplePSF(self, img,sigma,stride):
        def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride,stride),sigma)
        if img.ndim == 3:
            img_w,img_h,img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w,img_h = img.shape
            img = img.reshape((img_w,img_h,1))
        from scipy import signal
        out_img = np.zeros((img_w//(stride), img_h//(stride), img_c))
        for i in range(img_c):
            out = signal.convolve2d(img[:,:,i],h,'valid')  
            out_img[:,:,i] = out[::stride,::stride]
        return out_img

    def generate_low_HSI(self, img, scale_factor):
        (h, w, c) = img.shape
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        return img_lr 

    def generate_MSI(self, img, sp_matrix):
        w,h,c = img.shape
        self.msi_channels = sp_matrix.shape[1]
        if sp_matrix.shape[0] == c:
            img_msi = np.dot(img.reshape(w*h,c), sp_matrix).reshape(w,h,sp_matrix.shape[1])
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        return img_msi
    def __getitem__(self, index):
        img_patch = self.img_patch_list[index]
        img_lr = self.img_lr_list[index]

        img_msi = self.img_msi_list[index]

        img_lrmsi_frommsi = self.img_lrmsi_frommsi_list[index]
        img_lrmsi_fromlrhsi = self.img_lrmsi_fromlrhsi_list[index]

        img_tensor_lr = torch.from_numpy(img_lr.transpose(2,0,1).copy()).float()
        img_tensor_hr = torch.from_numpy(img_patch.transpose(2,0,1).copy()).float()
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2,0,1).copy()).float()
        img_tensor_lrmsi_frommsi = torch.from_numpy(img_lrmsi_frommsi.transpose(2,0,1).copy()).float()
        img_tensor_lrmsi_fromlrhsi = torch.from_numpy(img_lrmsi_fromlrhsi.transpose(2,0,1).copy()).float()
        
        return {"lhsi":img_tensor_lr,
                'hmsi':img_tensor_rgb,
                "hhsi":img_tensor_hr,
                'lrmsi_frommsi':img_tensor_lrmsi_frommsi,
                'lrmsi_fromlrhsi' :img_tensor_lrmsi_fromlrhsi
               }
        

    def __len__(self):
        return len(self.imgpath_list)
