import itertools
import tabnanny
import torch
import torch.utils.data as Torchdata
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from scipy import io
import os
from tools import *
from net import *
import time

# Parameters setting
DATASET = 'PaviaU'  #PaviaU、Salinas、HHK、Berlin、Houston2018
SAMPLE_ALREADY = False
N_RUNS = 10
SAMPLE_SIZE = 5
BATCH_SIZE_PER_CLASS =  SAMPLE_SIZE // 2
PATCH_SIZE = 15
FLIP_ARGUMENT = False
ROTATED_ARGUMENT = False
ITER_NUM = 1000
SAMPLING_MODE = 'fixed_withone'
FOLDER = './Datasets/'
LEARNING_RATE = 0.001
FEATURE_DIM = 64
GPU = 0
RESULT_DIR = './Results/'
numComponents = 10
k = 5
training_time = []

#############################Data###################################
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, numComponents, FOLDER)
N_CLASSES = len(LABEL_VALUES) - 1
N_BANDS = img.shape[-1]
def data_init(run):
    if SAMPLE_ALREADY:
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run, ITER_NUM)
    else:
        train_gt, test_gt, _, _ = sample_gt(gt, SAMPLE_SIZE, mode=SAMPLING_MODE)
        save_sample(train_gt, test_gt, DATASET, SAMPLE_SIZE, run, SAMPLE_SIZE, ITER_NUM)
    task_train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
    task_test_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),np.count_nonzero(gt)))
    return train_gt,task_train_dataset,task_test_dataset
#############################Optim##################################
criterion = nn.MSELoss().cuda(GPU)
#############################Train##################################
def train_run(run):
    print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
    display_iter = 10
    losses = np.zeros(ITER_NUM+1)
    mean_losses = np.zeros(ITER_NUM+1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    module_1 = DM_MRN_train(N_BANDS, FEATURE_DIM, PATCH_SIZE,numComponents,BATCH_SIZE_PER_CLASS,N_CLASSES,SAMPLE_SIZE,k)
    module_1.apply(weights_init)
    module_1.to(device)
    optimizer1 = torch.optim.Adam(module_1.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    scheduler1 = StepLR(optimizer1, step_size=ITER_NUM // 2, gamma=0.1)

    train_gt,task_train_dataset,task_test_dataset  = data_init(run)

    start_time = time.time()

    for iter_ in tqdm(range(1, ITER_NUM + 1), desc='Training the network'):
        task_test_gt, rest_gt, _, _ = sample_gt(train_gt, 1, mode='fixed_withone')
        task_train_gt, rest_gt, task_index, special_list = sample_gt(train_gt, BATCH_SIZE_PER_CLASS,mode='fixed_withone')
        task_test_dataset.resetGt(task_test_gt)
        task_test_loader = Torchdata.DataLoader(task_test_dataset, batch_size=N_CLASSES, shuffle=False)
        task_train_dataset.resetGt(task_train_gt)
        task_batch_size = np.count_nonzero(task_train_gt)
        task_train_loader = Torchdata.DataLoader(task_train_dataset, batch_size=task_batch_size, shuffle=False)

        samples, sample_labels = task_train_loader.__iter__().next()
        batches, batch_labels = task_test_loader.__iter__().next()

        module_1.train()
        result = module_1(samples.cuda(GPU),batches.cuda(GPU),task_index,special_list,iter_)
        one_hot_labels = torch.zeros(task_batch_size, N_CLASSES).scatter_(1,sample_labels.view(-1, 1),1).cuda(GPU)

        l_fr,l_stage,l_fused = cal_loss(criterion,result,one_hot_labels)
        loss = l_fr + l_stage + l_fused
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        scheduler1.step()
        losses[iter_] = loss.item()

        mean_losses[iter_] = np.mean(losses[max(0, iter_ - 10):iter_ + 1])
        if display_iter and iter_ % display_iter == 0:
            string = 'Train (ITER_NUM {}/{})\tLoss: {:.6f}'
            string = string.format(iter_, ITER_NUM, mean_losses[iter_])
            tqdm.write(string)


        # save networks
        save_module1 = 'Module1'
        with torch.cuda.device(GPU):
            if iter_ == ITER_NUM:
                model_module1_dir = RESULT_DIR + 'Checkpoints/' + save_module1 + '/' + task_train_loader.dataset.name + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
                if not os.path.isdir(model_module1_dir):
                    os.makedirs(model_module1_dir)
                model_module1_file = model_module1_dir + 'sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
                torch.save(module_1.state_dict(), model_module1_file)

    end_time = time.time()
    training_time.append(end_time - start_time)

    loss_dir = RESULT_DIR + 'Losses/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    loss_file = loss_dir + '/' + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '_dim' + str(FEATURE_DIM) +'.mat'
    io.savemat(loss_file, {'losses':losses})
####################################################################
def main():
    for run in range(N_RUNS):
        train_run(run)
    #record time
    ACCU_DIR = RESULT_DIR + 'Accu_File/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
    if not os.path.isdir(ACCU_DIR):
        os.makedirs(ACCU_DIR)
    time_path = ACCU_DIR+str(DATASET)+'_'+str(SAMPLE_SIZE)+'.txt'
    f = open(time_path, 'a')
    sentence0 = 'avergare training time:' + str(np.mean(training_time)) + '\n'
    f.write(sentence0)
    sentence1 = 'training time for each iteration are:' + str(training_time) + '\n'
    f.write(sentence1)
    f.close()

if __name__ == '__main__':
    main()     


