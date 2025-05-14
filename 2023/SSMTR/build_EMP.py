# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:28:40 2019

@author: HLB
"""

# Building the Extended Morphological Profiles (EMP)
from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util
import numpy as np
import skimage.morphology as sm
def opening_by_reconstruction(image, se):
    """
        Performs an Opening by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed


def closing_by_reconstruction(image, se):
    """
        Performs a Closing by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(
        obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted


def build_morphological_profiles(image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the morphological profiles for a given image.

        Parameters:
            base_image: 2d matrix, it is the spectral information part of the MP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns: 
            emp: 3d matrix with both spectral (from the base_image) and spatial information         
    """
    x, y = image.shape

    cbr = np.zeros(shape=(x, y, num_openings_closings))
    obr = np.zeros(shape=(x, y, num_openings_closings))

    it = 0
    tam = se_size
    while it < num_openings_closings:
        se = disk(tam)
        temp = closing_by_reconstruction(image, se)
        cbr[:, :, it] = temp[:, :]
        temp = opening_by_reconstruction(image, se)
        obr[:, :, it] = temp[:, :]
        tam += se_size_increment
        it += 1

    mp = np.zeros(shape=(x, y, (num_openings_closings*2)+1))
    cont = num_openings_closings - 1
    for i in range(num_openings_closings):
        mp[:, :, i] = cbr[:, :, cont]
        cont = cont - 1

    mp[:, :, num_openings_closings] = image[:, :]

    cont = 0
    for i in range(num_openings_closings+1, num_openings_closings*2+1):
        mp[:, :, i] = obr[:, :, cont]
        cont += 1

    return mp


def build_emp(base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the extended morphological profiles for a given set of images.

        Parameters:
            base_image: 3d matrix, each 'channel' is considered for applying the morphological profile. It is the spectral information part of the EMP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    base_image_rows, base_image_columns, base_image_channels = base_image.shape
    se_size = se_size
    se_size_increment = se_size_increment
    num_openings_closings = num_openings_closings
    morphological_profile_size = (num_openings_closings * 2) + 1
    emp_size = morphological_profile_size * base_image_channels
    emp = np.zeros(
        shape=(base_image_rows, base_image_columns, emp_size))

    cont = 0
    for i in range(base_image_channels):
        # build MPs
        mp_temp = build_morphological_profiles(
            base_image[:, :, i], se_size, se_size_increment, num_openings_closings)

        aux = morphological_profile_size * (i+1)

        # build the EMP
        cont_aux = 0
        for k in range(cont, aux):
            emp[:, :, k] = mp_temp[:, :, cont_aux]
            cont_aux += 1

        cont = morphological_profile_size * (i+1)

    return emp
def build_emp1(base_image, nScale=3):
    row = base_image.shape[0]
    col = base_image.shape[1]
    nPC = base_image.shape[2]
    emp = np.zeros([row, col,2*nScale*nPC])
    i = 0
    for iScale in range(nScale):
        for iPC in range(nPC):    				
            se = sm.disk( 3 * (iScale + 1))   				
            x=sm.opening(base_image[:, : , iPC],se)    				
            y=sm.closing(base_image[:, :, iPC ],se)
            emp[:,:,i] = x
            emp[:,:,i+1] = y
            i = i + 2
    return emp
                 
    
    
    
