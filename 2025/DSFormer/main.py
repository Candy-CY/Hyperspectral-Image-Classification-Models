import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset , sample_gt_by_class
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test
from utils.HyperTools import *
import os
import time
from sklearn.decomposition import PCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='DSFormer') # model name
    parser.add_argument("--dataset_name", type=str, default="pu",
                        choices=['pu','sa','ip','whulk','whuhc','whuhh','houston13','houston18']) # dataset name
    parser.add_argument("--dataset_dir", type=str, default="./datasets") # dataset dir
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=10) # crop patch_size
    parser.add_argument("--ps", type=int, default=2) # vit patch_size
    parser.add_argument("--num_run", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--bs", type=int, default=256)  # bs = batch size
    parser.add_argument("--train_num", type=int, default=30)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--k', type=str, default='1/2')

    parser.add_argument('--use_pca', type=bool, default=True)
    parser.add_argument("--pc", type=int, default=30)

    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--group_num", type=int, default=4)
    # parser.add_argument('--save_exp_name', type=str, default='_50/')
    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))

    DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Houston', 4: 'IndianP',
                5: 'LongKou', 6: 'Hanchuan', 7: 'HongHu', 8: 'Houston18'}

    # save dir path
    save_exp_name = str(opts.train_num)
    save_path_prefix = opts.save_path_prefix + 'Exp_' + DataName[opts.dataID] + '_' + save_exp_name + '/'
    log_path_prefix = opts.save_path_prefix + 'Log_' + DataName[opts.dataID] + '_' + save_exp_name + '/'


    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)
    if os.path.exists(log_path_prefix) == False:
        os.makedirs(log_path_prefix)
    time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())

    log_file = log_path_prefix + opts.model + '_' + time_current +'.txt'
    opts.k = eval(opts.k)

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
    X = torch.tensor(image).unsqueeze(0).permute(0,3,1,2)
    num_classes = len(labels)
    num = num_classes * opts.train_num
    num_bands = image.shape[-1]

    # random seeds
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 10001, 202409, 202410]

    # empty list to storing results
    results = []
    train_times = []
    test_times = []
    times = []

    for run in range(opts.num_run):
        start_time = time.time()
        start_time_train = time.time()

        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run + 1, opts.num_run))

        # get train_gt, and test_gt
        train_gt, test_gt = sample_gt_by_class(gt, opts.train_num, seeds[run])
        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True, use_pca=opts.use_pca)
        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)

        # load model and loss
        model = get_model(opts.model, opts.dataset_name,kernel_size=opts.kernel_size, ps=opts.ps, k= opts.k,group_num=opts.group_num,emb_dim=opts.emb_dim)

        if run == 0:
            split_info_print(train_gt, test_gt, labels)


        model = model.to(device)
        optimizer, scheduler = load_scheduler(opts.model, model)

        criterion = nn.CrossEntropyLoss()

        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        try:
            train(model, optimizer, criterion, train_loader, train_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')
        end_time_train = time.time()
        train_times.append(end_time_train - start_time_train)  # train time

        # test the model
        if opts.use_pca==True:
            original_shape = image.shape
            reshaped_image = image.reshape(-1, original_shape[2])
            pca = PCA(n_components=opts.pc)
            pca_result = pca.fit_transform(reshaped_image)
            image = pca_result.reshape(original_shape[0], original_shape[1], opts.pc)

        start_time_test = time.time()
        # probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device, X)
        probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
        prediction = np.argmax(probabilities, axis=-1)

        end_time_test = time.time()
        test_times.append(end_time_test-start_time_test)   # test time

        end_time = time.time()
        times.append(end_time-start_time)
        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
        results.append(run_results)
        show_results(run_results, label_values=labels)
        OA = run_results["Accuracy"]
        kappa = run_results["Kappa"]

        # Draw result
        img = DrawResult(np.reshape(prediction+1, -1), opts.dataID)  # flatten
        plt.imsave(save_path_prefix + opts.model + '_OA' + repr(int(OA * 10000)) + '_kappa' + repr(
            int(kappa * 10000)) +  '_PS' + repr(opts.patch_size)+'.png', img)  # save

        # end_time = time.time()
        # times.append(end_time - start_time)
    if opts.num_run > 1:
        avg_time_train = np.mean(train_times)
        std_time_train = np.std(train_times)

        avg_time_test = np.mean(test_times)
        std_time_test = np.std(test_times)

        avg_time = np.mean(times)
        std_time = np.std(times)

        text=show_results(results, label_values=labels, agregated=True,file_path=log_file, avg_time_train=avg_time_train,
                          std_time_train=std_time_train, avg_time_test=avg_time_test, std_time_test=std_time_test,
                          avg_time = avg_time, std_time=std_time, opts = opts)

