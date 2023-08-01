import torch
import torch.utils.data as Torchdata
import numpy as np
from tqdm import tqdm
from tools import *
from net import *
import scipy.io as sio
import time


# Parameters setting
DATASET = 'PaviaU'  #PaviaU、Salinas、HHK、Berlin、Houston2018
N_RUNS = 10
SAMPLE_SIZE = 5
PATCH_SIZE = 15
FOLDER = './Datasets/'
PRE_ALL = False
FEATURE_DIM = 64
GPU = 0
ITER_NUM = 1000
BATCH_SIZE_PER_CLASS =  SAMPLE_SIZE // 2
DRAW_OR_NOT = True
RESULT_DIR = './Results/'
ACCU_DIR = RESULT_DIR + 'Accu_File/' + DATASET + '/'+ str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
CHECKPOINT_Module1 = RESULT_DIR + 'Checkpoints/Module1/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
numComponents = 10
k = 5
testing_time = []

##############################Data##################################
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, numComponents, FOLDER)
N_CLASSES = len(LABEL_VALUES) - 1
N_BANDS = img.shape[-1]
def data_init(run):
    train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run, ITER_NUM)
    task_index = []
    special_list = []
    for c in np.unique(train_gt):
        if c == 0:
            continue
        X = np.count_nonzero(train_gt == c)
        if X < SAMPLE_SIZE:
            task_index.append(c)
            special_list.append(X)
    if PRE_ALL:
        test_gt = np.ones_like(test_gt)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
    test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
    return train_dataset, test_dataset, task_index, special_list, np.count_nonzero(train_gt)
##############################Test##################################
def test_run(run):
    print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
    oa=[]

    train_dataset, test_dataset ,task_index,special_list,loader_size = data_init(run)
    train_loader = Torchdata.DataLoader(train_dataset, batch_size=loader_size, shuffle=False)
    test_loader = Torchdata.DataLoader(test_dataset, batch_size=64, shuffle=False)
    tr_data, tr_labels = train_loader.__iter__().next()
    tr_data = tr_data.cuda(GPU)

    module_1 = DM_MRN_test(N_BANDS, FEATURE_DIM, PATCH_SIZE,numComponents,BATCH_SIZE_PER_CLASS, N_CLASSES, SAMPLE_SIZE,k)

    if CHECKPOINT_Module1 is not None:
        Module_file1 = CHECKPOINT_Module1 + 'sample{}_run{}.pth'.format(SAMPLE_SIZE, run)
        with torch.cuda.device(GPU):
            module_1.load_state_dict(torch.load(Module_file1))
    else:
        raise ('No Chenkpoints for Module_1 Net')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module_1.to(device)

    module_1.eval()
    test_labels = []
    pre_labels = []
    pad_pre_gt = np.zeros_like(test_dataset.label)
    pad_test_indices = test_dataset.indices
    start_time = time.time()
    for batch_idx, (te_data, te_labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
        with torch.no_grad():
            te_data, te_labels = te_data.cuda(GPU), te_labels.cuda(GPU)
            N_TEST = len(te_labels)
            result = module_1(tr_data,te_data,task_index,special_list,N_TEST)
            result = torch.split(result, 1, dim=1)
            for i in range(len(result)):
                relation_i = result[i]
                c = 0
                for j in range(N_CLASSES):
                    if j + 1 in task_index:
                        y_front = relation_i[:j * SAMPLE_SIZE + special_list[c], :]
                        y_back = relation_i[j * SAMPLE_SIZE + special_list[c]:, :]
                        remain = SAMPLE_SIZE - special_list[c]
                        y = torch.zeros((remain, 1)).cuda(0)
                        relation_i = [y_front, y, y_back]
                        relation_i = torch.cat(relation_i, dim=0)
                        c = c + 1
                    else:
                        pass
                relation_i = relation_i.view(-1, SAMPLE_SIZE)
                scores, _ = torch.max(relation_i, dim=1)
                _, output = torch.max(scores, dim=0)
                pre_labels.append(output.tolist())
            test_labels.extend(te_labels.tolist())
    end_time = time.time()
    for i in range(len(pad_test_indices)):
        pad_pre_gt[pad_test_indices[i]] = pre_labels[i] + 1
    p = PATCH_SIZE // 2
    pad_pre_gt = pad_pre_gt[p:-p, p:-p]
    if not PRE_ALL:
        accuracy, total = 0, 0
        for iten_i in range(len(pre_labels)):
            accuracy += test_labels[iten_i] == pre_labels[iten_i]
            total += 1
        rate = accuracy / total
        oa.append(rate)
        print('Accuracy:', rate)
    else:
        mask = np.zeros_like(gt)
        mask[np.where(gt != 0)] = 1
        pre_gt_label = pad_pre_gt * mask
        gt_label_f = gt[np.where(gt != 0)].flatten()
        pre_gt_label_f = pre_gt_label[np.where(pre_gt_label != 0)].flatten()
        accuracy = np.zeros_like(gt_label_f)
        accuracy[np.where(gt_label_f == pre_gt_label_f)] = 1
        rate = np.sum(accuracy) / gt_label_f.size
        oa.append(rate)
        print('Accuracy:', rate)

    testing_time.append(end_time - start_time)
    print('Testing Time:', end_time - start_time)

    #save sores
    results = dict()
    results['OA'] = rate
    if PRE_ALL:
        results['pre_all'] = np.asarray(pad_pre_gt,dtype='uint8')
        results['pre_gt'] = np.asarray(pre_gt_label, dtype='uint8')
    else:
        results['pre_gt'] = np.asarray(pad_pre_gt, dtype='uint8')
    results['test_labels'] = test_labels
    results['pre_labels'] = pre_labels
    save_result(results, ACCU_DIR, SAMPLE_SIZE, run)
####################################################################
def main():
    for run in range(N_RUNS):
        test_run(run)
        if DRAW_OR_NOT:
            gt = get_gt(DATASET, FOLDER)
            gt = gt.tolist()
            if PRE_ALL:
                result = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '.mat')['pre_all']
            else:
                result = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '.mat')['pre_gt']
            result = result.tolist()
            pic_dir = RESULT_DIR + 'Pic/' + DATASET + '/'+ str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
            if not os.path.isdir(pic_dir):
                os.makedirs(pic_dir)
            drawresult(result, DATASET, str(SAMPLE_SIZE), str(ITER_NUM), gt, pic_dir, run)
    matrix(ACCU_DIR, DATASET, SAMPLE_SIZE, ITER_NUM, N_RUNS, testing_time)
####################################################################
if __name__ == '__main__':
    main()     

