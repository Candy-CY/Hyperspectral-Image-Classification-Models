from scipy import io
import os
import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data
import random
from sklearn.decomposition import PCA

def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'pu',
        'sa',
        'ip',
        'whulk',
        'whuhc',
        'whuhh',
        'houston13',
        'houston18',
        'QUH_TDW',
        'QUH_QY',
        'QUH_PA'
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'pu'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'ip'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_corrected.mat"))
        image = image['indian_pines_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_gt.mat"))
        gt = gt['indian_pines_gt']
        labels = [
            'Undefined',
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhh':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu.mat"))
        image = image['WHU_Hi_HongHu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu_gt.mat"))
        gt = gt['WHU_Hi_HongHu_gt']
        labels = [
            'Undefined',
            'Red roof',
            'Road',
            'Bare soil',
            'Cotton',
            'Cotton firewood',
            'Rape',
            'Chinese cabbage',
            'Pakchoi',
            'Cabbage',
            'Tuber mustard',
            'Brassica parachinensis',
            'Brassica chinensis',
            'Small Brassica chinensis',
            'Lactuca sativa',
            'Celtuce',
            'Film covered lettuce',
            'Romaine lettuce',
            'Carrot',
            'White radish',
            'Garlic sprout',
            'Broad bean',
            'Tree',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhc':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan.mat"))
        image = image['WHU_Hi_HanChuan']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan_gt.mat"))
        gt = gt['WHU_Hi_HanChuan_gt']
        labels = [
            'Undefined',
            'Strawberry',
            'Cowpea',
            'Soybean',
            'Sorghum',
            'Water spinach',
            'Watermelon',
            'Greens',
            'Trees',
            'Grass',
            'Red roof',
            'Gray roof',
            'Plastic',
            'Bare soil',
            'Road',
            'Bright object',
            'Water',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'houston13':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013.mat"))
        image = image['GRSS2013']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013_gt.mat"))
        gt = gt['GRSS2013_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'houston18':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Houston2018.mat"))
        image = image['Houston2018']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Houston2018_gt.mat"))
        gt = gt['Houston2018_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Artificial turf",
            "Evergreen trees",
            "Deciduous trees",
            "Bare earth",
            "Water",
            "Residential buildings",
            "Non-residential buildings",
            "Roads",
            "Sidewalks",
            "Crosswalks",
            "Major thoroughfares",
            "Highways",
            "Railways",
            "Paved parking lots",
            "Unpaved parking lots",
            "Cars",
            "Trains",
            "Stadium seats",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_TDW':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Tangdaowan.mat"))
        image = image['Tangdaowan']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Tangdaowan_GT.mat"))
        gt = gt['TangdaowanGT']
        labels = [
            "Undefined",
            "Rubber track",
            "Flagging",
            "Sandy",
            "Asphalt",
            "Boardwalk",
            "Rocky shallows",
            "Grassland",
            "Bulrush",
            "Gravel road",
            "Ligustrum vicaryi",
            "Coniferous pine",
            "Spiraea",
            "Bare soil",
            "Buxus sinica",
            "Photinia serrulata",
            "Populus",
            "Ulmus pumila L",
            "Seawater",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_QY':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Qingyun.mat"))
        image = image['Chengqu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Qingyun_GT.mat"))
        gt = gt['ChengquGT']
        labels = [
            "Undefined",
            "Trees",
            "Concrete building",
            "Car",
            "Ironhide building",
            "Plastic playground",
            "Asphalt road",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_PA':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Pingan.mat"))
        image = image['Haigang']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Pingan_GT.mat"))
        gt = gt['HaigangGT']
        labels = [
            "Undefined",
            "Ship",
            "Seawater",
            "Trees",
            "Concrete structure building",
            "Floating pier",
            "Brick houses",
            "Steel houses",
            "Wharf construction land",
            "Car",
            "Road",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0


    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 filter nan values out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # step2 normalise the HSI data (method from SSAN, TGRS 2020)
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    mean_by_c = np.mean(image, axis=(0, 1))
    for c in range(image.shape[-1]):
        image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    return image, gt, labels


def sample_gt(gt, percentage, seed):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return:
    """
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    train_indices, test_indices = sklearn.model_selection.train_test_split(
        X,
        train_size=percentage,
        random_state=seed,
        stratify=y
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    return train_gt, test_gt


def sample_gt_by_class(gt, n, seed=None):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param n: number of samples per class for training
    :param seed: random seed
    :return: train_gt, test_gt
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 找到所有非-1的元素的索引
    indices = np.where(gt >= 0)
    # 创建一个字典用于存储每个类别的索引
    class_indices = {}
    for idx in zip(*indices):
        label = gt[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # 创建与gt形状相同的数组，初始值为-1，分别表示训练集和测试集
    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    # 对每个类别进行随机选择
    for label, idx_list in class_indices.items():
        if len(idx_list) < n:
            # raise ValueError(f"Class {label} has less than {n} samples.")
            num_sample = 15
        else:
            num_sample = n
        selected_indices = random.sample(idx_list, num_sample)
        for idx in selected_indices:
            train_gt[idx] = gt[idx]
        for idx in idx_list:
            if idx not in selected_indices:
                test_gt[idx] = gt[idx]

    return train_gt, test_gt


def sampling_my(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], []

    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class]

        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15


        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, test_label_index_list, all_label_index_list


class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True,use_pca=False, pc=30):
        """
        :param image: 3d float np array of HSI, image
        :param gt: train_gt or val_gt or test_gt
        :param patch_size: 7 or 9 or 11 ...
        :param data_aug: whether to use data augment, default is True
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2  # padding size
        self.use_pca = use_pca
        self.pc = pc

        if self.use_pca:
            # Reshape image for PCA
            original_shape = image.shape
            reshaped_image = image.reshape(-1, original_shape[2])
            pca = PCA(n_components=self.pc)
            pca_result = pca.fit_transform(reshaped_image)
            image = pca_result.reshape(original_shape[0], original_shape[1], self.pc)
        self.data = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image.shape[0] + self.ps
                                 and self.ps <= y < image.shape[1] + self.ps])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def hsi_augment(self, data):
        # e.g. (7 7 200) data = numpy array float32
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            data = self.hsi_augment(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Add a fourth dimension for 3D CNN

        data = data.unsqueeze(0)

        return data, label

