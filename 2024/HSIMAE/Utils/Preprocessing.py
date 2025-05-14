import numpy as np
from tqdm import tqdm
from itertools import product

from Utils.GroupWisePCA import applyGWPCA


def get_inital_seq(length, size, stride):
    n1 = length // size
    l_r = length - n1 * size
    size_2 = int(size // stride)
    n2 = l_r // size_2
    l_rr = l_r - n2 * size_2
    if l_rr == 0:
        num = int((n1 - 1) * stride + n2 + 1)
    else:
        num = int((n1 - 1) * stride + n2 + 2)
    seq = np.arange(0, num * size_2, size_2)
    seq[-1] = length - size
    return seq


def splitHSI(data, target_size, stride, zero_cut=None, zeros_ref=None):
    w, h, c = data.shape
    ws, hs, cs = stride
    rowsize, colsize, chsize = target_size

    row_seq = get_inital_seq(w, rowsize, ws)
    col_seq = get_inital_seq(h, colsize, hs)
    ch_seq = get_inital_seq(c, chsize, cs)

    num = 0
    pieces = []
    zeros = 0
    count_zeros = 0
    for ch in ch_seq:
        one_num = 0
        for r in row_seq:
            for c in col_seq:
                piece = data[r: r + rowsize, c: c + colsize, ch: ch + chsize]
                if zero_cut is not None:
                    if (- target_size[0] <= r - zeros_ref[0] <= zero_cut[0]) and \
                            (- target_size[1] <= c - zeros_ref[1] <= zero_cut[1]):
                        zeros += 1
                    else:
                        if np.min(np.sum(np.abs(piece), 2)) == 0:
                            count_zeros += 1
                        pieces.append(piece)
                        num += 1
                        one_num += 1
                else:
                    pieces.append(piece)
                    num += 1
                    one_num += 1
    return pieces, one_num, len(ch_seq)


def get_cut_loc(index, cut_locs):
    ch_seq, row_seq, col_seq = cut_locs
    ch_index = index // (len(row_seq) * col_seq)
    r_index = index % (len(row_seq) * col_seq)
    c_index = index % (len(col_seq))
    ch = ch_seq[ch_index]
    r = row_seq[r_index]
    c = col_seq[c_index]
    return [r, c, ch]


def get_split_info(data, target_size, stride, num, max, min):
    w, h, c = data.shape
    ws, hs, cs = stride
    rowsize, colsize, chsize = target_size

    ch_seq = get_inital_seq(c, chsize, cs)
    row_seq = get_inital_seq(w, rowsize, ws)
    col_seq = get_inital_seq(h, colsize, hs)

    cut_locs = list(product(ch_seq, row_seq, col_seq, [num], [max], [min]))
    return cut_locs


def get_data_cut_file(data_path, patch_size=9, save_path=None, norm=False, GWPCA=True, ratio=1.0):
    data_cubes = []
    cut_locs = []

    num_count = 0
    for path in tqdm(data_path):
        HSI_data = np.load(path)

        if GWPCA:
            HSI_data = applyGWPCA(HSI_data, nc=32, group=4, whiten=True)
        w, h, c = HSI_data.shape

        if norm:
            max_ = np.max(HSI_data)
            min_ = np.min(HSI_data)
        else:
            max_ = 1
            min_ = 0

        if num_count >= 14:
            cut_loc = get_split_info(HSI_data, (9, 9, c), (1, 1, 1), num_count, max_, min_)
        else:
            print(path)
            cut_loc = get_split_info(HSI_data, (9, 9, c), (3, 3, 1), num_count, max_, min_)
            cut_loc = np.array(cut_loc)
            np.random.shuffle(cut_loc)
            new_length = int(cut_loc.shape[0] * ratio)
            cut_loc = list(cut_loc[:new_length])
            print('cut num: ', len(cut_loc))
        cut_locs += cut_loc
        num_count += 1
        data_cubes.append(HSI_data)
    cut_locs = np.array(cut_locs, dtype=np.int16)
    if save_path:
        np.save(save_path, cut_locs)
    return [data_cubes, cut_locs]


def get_data_set(data, gt_path, patch_size=9, percent=None, num=None, mask=None):
    pad = patch_size // 2
    if patch_size % 2 == 0:
        HSI_data_pad = np.pad(data, ((pad, pad - 1), (pad, pad - 1), (0, 0)), 'reflect')
    else:
        HSI_data_pad = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')

    w, h, c = HSI_data_pad.shape

    data_cubes, one_cut_num, ch_num = splitHSI(HSI_data_pad, (patch_size, patch_size, c), (patch_size, patch_size, 1))

    gt_raw = np.load(gt_path)
    gt = gt_raw.reshape(-1)
    assert len(data_cubes) == gt.shape[0]

    n_classes = len(np.unique(gt))
    assert n_classes == (gt.max() + 1)
    class_num_count = np.zeros(n_classes)

    train_data = []
    train_labels = []

    test_num = 0
    test_gt = gt.copy()

    if mask is not None:
        mask = np.load(mask)
        mask = mask.reshape(-1)
        assert len(mask) == len(gt)

        for i, ind in tqdm(enumerate(mask)):
            if ind == 0:
                test_num += 1
            else:
                class_num_count[ind] += 1
                train_data.append(data_cubes[i])
                train_labels.append(ind)
                test_gt[i] = 0
    else:
        indices = np.arange(gt.shape[0])
        shuffled_indices = np.random.permutation(indices)
        images = [data_cubes[i] for i in shuffled_indices]
        labels = gt[shuffled_indices]

        num_per_class = np.array([np.sum(labels == l) for l in range(n_classes)])
        if percent:
            train_num_per_class = np.ceil(num_per_class * percent)
        elif num:
            train_num_per_class = np.zeros(n_classes) + num
            b = np.bincount(gt)
            for i, c in enumerate(b):
                if c == num:
                    train_num_per_class[i] = num - 5
        else:
            return

        for i, ind in tqdm(enumerate(labels)):
            if ind == 0:
                test_num += 1
            else:
                class_num_count[ind] += 1
                if class_num_count[ind] <= train_num_per_class[ind]:
                    train_data.append(images[i])
                    train_labels.append(ind)
                    test_gt[shuffled_indices[i]] = 0
    train_labels = np.array(train_labels)
    return train_data, train_labels, data_cubes, test_gt, gt_raw


def get_data_set_dual(data_path, gt_path=None, patch_size=9, percent=None, num=None, mask=None, norm=False, GWPCA=True):
    HSI_data_raw = np.load(data_path)

    if GWPCA:
        HSI_data_raw = applyGWPCA(HSI_data_raw, nc=32, group=4, whiten=True)

    if norm:
        max_ = np.max(HSI_data_raw)
        min_ = np.min(HSI_data_raw)
        HSI_data = (HSI_data_raw - min_) / (max_ - min_)
    else:
        HSI_data = HSI_data_raw

    w, h, c = HSI_data.shape
    print(HSI_data.shape)

    data_cubes_2, one_cut_num, ch_num = splitHSI(HSI_data, (patch_size, patch_size, c), (1, 1, 1))
    data_cubes_2 = np.array(data_cubes_2)

    pad = patch_size // 2
    HSI_data_pad = np.pad(HSI_data, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
    w, h, c = HSI_data_pad.shape

    data_cubes, one_cut_num, ch_num = splitHSI(HSI_data_pad, (patch_size, patch_size, c), (patch_size, patch_size, 1))
    data_cubes = np.array(data_cubes)

    gt_raw = np.load(gt_path)
    gt = gt_raw.reshape(-1)
    assert len(data_cubes) == gt.shape[0]

    n_classes = len(np.unique(gt))
    assert n_classes == (gt.max() + 1)
    class_num_count = np.zeros(n_classes)

    train_index = []
    test_index = []

    test_num = 0
    test_gt = gt.copy()

    if mask is not None:
        mask = np.load(mask)
        mask = mask.reshape(-1)
        assert len(mask) == len(gt)

        for i, ind in tqdm(enumerate(mask)):
            if ind == 0:
                test_num += 1
                test_index.append(i)
            else:
                class_num_count[ind] += 1
                train_index.append(i)
                test_gt[i] = 0
    else:
        indices = np.arange(gt.shape[0])
        shuffled_indices = np.random.permutation(indices)
        labels = gt[shuffled_indices]

        num_per_class = np.array([np.sum(labels == l) for l in range(n_classes)])
        if percent:
            train_num_per_class = np.ceil(num_per_class * percent)
        elif num:
            train_num_per_class = np.zeros(n_classes) + num
            b = np.bincount(gt)
            for i, cls in enumerate(b):
                if cls == num:
                    train_num_per_class[i] = num - 5
        else:
            return

        for i, ind in tqdm(enumerate(labels)):
            if ind == 0:
                test_num += 1
                test_index.append(shuffled_indices[i])
            else:
                class_num_count[ind] += 1
                if class_num_count[ind] <= train_num_per_class[ind]:
                    train_index.append(shuffled_indices[i])
                    test_gt[shuffled_indices[i]] = 0
                else:
                    test_index.append(shuffled_indices[i])

    train_labels = gt[train_index]
    test_gt = test_gt.reshape(gt_raw.shape)
    return train_index, train_labels, data_cubes_2, data_cubes, test_gt, gt_raw


def spilt_dataset(data, label, training_ratio=0.8):
    indices = np.arange(label.shape[0])
    shuffled_indices = np.random.permutation(indices)
    label_sf = label[shuffled_indices]

    n_classes = len(np.unique(label))
    assert n_classes == label.max()

    num_per_class = np.array([np.sum(label == l + 1) for l in range(n_classes)])
    val_num_per_class = num_per_class * (1 - training_ratio)
    class_num_count = np.zeros(n_classes)

    train_index = []
    val_index = []

    for i, ind in tqdm(enumerate(label_sf)):
        index = ind - 1
        class_num_count[index] += 1
        if class_num_count[index] <= val_num_per_class[index]:
            val_index.append(shuffled_indices[i])
        else:
            train_index.append(shuffled_indices[i])
    if training_ratio == 1:
        val_index = train_index[:int(len(train_index) * 0.2)]
    return [data[i] for i in train_index], label[train_index], [data[i] for i in val_index], label[val_index]