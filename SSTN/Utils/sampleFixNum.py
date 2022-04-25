import numpy as np

def samplingFixedNum_TrainTestEqual(sample_num, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train_ = {}
    test_ = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[-sample_num:]
    train_fix_indices = []
    test_fix_indices = []
    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]
    print train_fix_indices
    print test_fix_indices
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    return  train_fix_indices, test_fix_indices

def samplingFixedNum(sample_num, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train_ = {}
    test_ = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[sample_num:]                     #difference derivation
    train_fix_indices = []
    test_fix_indices = []
    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    return  train_fix_indices, test_fix_indices

def samplingDiffFixedNum(sample_num_list, groundTruth):
    labels_loc = {}
    train_ = {}
    test_ = {}
    for idx, each_num in enumerate(sample_num_list):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == idx + 1]
        np.random.shuffle(indices)
        labels_loc[idx] = indices
        train_[idx] = indices[:each_num]
        test_[idx] = indices[each_num:]
    train_diff_indices = []
    test_diff_indices = []
    for i in range(len(sample_num_list)):
        train_diff_indices += train_[i]
        test_diff_indices += test_[i]
    np.random.shuffle(train_diff_indices)
    np.random.shuffle(test_diff_indices)
    return train_diff_indices, test_diff_indices

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    print train_indices, test_indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices