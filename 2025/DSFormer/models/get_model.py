from .DSFormer import *
from .cnn2d import *

def get_model(model_name, dataset_name, kernel_size, ps, k, group_num, emb_dim):
    if model_name == 'DSFormer':
        model = DSFormer(dataset_name,kernel_size,ps, k, group_num,emb_dim)

    elif model_name == 'cnn2d':
        model = cnn2d(dataset_name)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model

