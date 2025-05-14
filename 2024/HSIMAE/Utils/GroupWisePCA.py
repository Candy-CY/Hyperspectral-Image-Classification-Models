import numpy as np
from sklearn.decomposition import PCA


def split_data(data_list, group=4):
    output_data = data_list
    step = group // 2
    for i in range(step):
        split_data = []
        for data in output_data:
            n, c = data.shape
            data_s1 = data[:, :c // 2]
            data_s2 = data[:, c // 2:]
            split_data.append(data_s1)
            split_data.append(data_s2)
        output_data = split_data
    return output_data


def applyGWPCA(X, nc=32, group=4, whiten=True):
    h, w, c = X.shape
    X = np.reshape(X, (-1, c))
    X = (X - X.min()) / (np.max(X) - np.min(X))

    X_split = split_data([X], group)
    pca_data_list = []
    for i, x in enumerate(X_split):
        pca = PCA(n_components=nc // group, whiten=whiten, random_state=42)
        pca_data = pca.fit_transform(x)
        pca_data_list.append(pca_data)

    out = np.concatenate(pca_data_list, axis=-1)
    out = np.reshape(out, (h, w, -1))
    return out