import numpy as np


def label_to_colormap(label):
    assert label.max() < 20, 'only 20 classes are supported'
    label_to_color_dict = {
        0: [0, 0, 0],
        1: [128, 0, 0],
        2: [0, 128, 0],
        3: [128, 128, 0],
        4: [0, 0, 128],
        5: [128, 0, 128],
        6: [0, 128, 128],
        7: [0, 64, 128],
        8: [64, 0, 0],
        9: [192, 0, 0],
        10: [64, 128, 0],
        11: [192, 128, 0],
        12: [64, 0, 128],
        13: [192, 0, 128],
        14: [64, 128, 128],
        15: [192, 128, 128],
        16: [0, 64, 0],
        17: [128, 64, 0],
        18: [0, 192, 0],
        19: [128, 192, 0],
    }

    color_map = np.zeros((label.shape[0], label.shape[1], 3))
    for i in np.unique(label):
        color_map[label == i] = label_to_color_dict[i]
    return color_map.astype(np.uint8)