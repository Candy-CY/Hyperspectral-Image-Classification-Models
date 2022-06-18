import numpy as np

def Normalization(raw_data):
    MAX = np.max(raw_data.ravel()).astype('float32')
    MIN = np.min(raw_data.ravel())
    new_data = (raw_data - MIN)/(MAX - MIN)- 0.5
    return new_data