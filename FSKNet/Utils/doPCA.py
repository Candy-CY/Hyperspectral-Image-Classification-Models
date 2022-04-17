from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def dimension_PCA(data, data_UP, input_dimension):
    pca = PCA(n_components=input_dimension)
    pca.fit(data)

    whole_pca = np.zeros((data_UP.shape[0], data_UP.shape[1], input_dimension))
    print (whole_pca.shape)

    for i in range(input_dimension):
         whole_pca[:, :, i] = pca.components_[i].reshape(data_UP.shape[0], data_UP.shape[1])

    print (whole_pca.shape)

    return whole_pca

