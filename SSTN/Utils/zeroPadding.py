import numpy as np

def zeroPadding_2D(old_matrix, pad_length):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length),(pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

#def zeroPadding_1D(old_vector, pad_length):


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def zeroPadding_1D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((0, pad_length)), 'constant', constant_values=0)
    return new_matrix

