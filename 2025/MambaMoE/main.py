import os
import time
import argparse
import torch
from torch.autograd import Variable
from HyperTools import *
import logging
import utils_logger
from tqdm import tqdm
import seaborn as sns
from Models.Model_S3ANet import *
from Models.MambaMoE import *

from Tools.utils import metrics, show_results
from Tools.dataset import load_mat_hsi

DataName = {1: 'PaviaU', 3: 'Houston', 6:'Hanchuan'}

def main(args):
    if args.dataID == 1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/PaviaU_15/'
    elif args.dataID == 3:
        num_classes = 15
        num_features = 144
        save_pre_dir = './Data/Houston_15/'
    elif args.dataID == 6:
        num_classes = 16
        num_features = 274
        save_pre_dir = './Data/Whuhi_hanchuan_30/'

    _, _, label_value = load_mat_hsi(args.dataset_name, args.dataset_dir)    # label value
    X = np.load(save_pre_dir + 'X.npy')
    _, h, w = X.shape
    Y = np.load(save_pre_dir + 'Y.npy')
    args.input_size = [h,w]

    X_train = np.reshape(X, (1, num_features, h, w))
    train_array = np.load(save_pre_dir + 'train_array.npy')
    test_array = np.load(save_pre_dir + 'test_array.npy')
    Y_train = np.ones(Y.shape) * 255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train, (1, h, w))

    # save dir path
    save_exp_name = str(args.train_num)   # train_num -> save path
    save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '_' + save_exp_name + '/'
    log_path_prefix = args.save_path_prefix + 'Log_' + DataName[args.dataID] + '_' + save_exp_name + '/'

    save_path_visual = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '_' + save_exp_name + '_visual/'

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)
    if os.path.exists(log_path_prefix) == False:
        os.makedirs(log_path_prefix)
    if os.path.exists(save_path_visual) == False:
        os.makedirs(save_path_visual)

    time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())
    log_file = log_path_prefix + args.model + '_' + time_current + '.txt'

    config = dict(
        in_channels=103,
        num_classes=9,
        block_channels=(96, 128, 192, 256),
        num_blocks=(1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )

    seed_list = [202501,202502,202503,202504,202505,202506,202507,202508,202509,2025010]

    results = []
    train_times = []
    test_times = []
    times = []

    for run in range(args.num_run):
        start_time = time.time()
        start_time_train = time.time()

        np.random.seed(seed_list[run])
        print("running an experiment with the {} model".format(args.model))
        print("run {} / {}".format(run + 1, args.num_run))
        # model
        if args.model == 'MambaMoE':
            Model = MambaMoE(num_features=num_features, num_classes=num_classes, embed_dim=args.emb_dim, patch_size=args.patch_size)
            num_epochs = args.epoch
        elif args.model == 'S3ANet':
            Model = S3ANet(num_features=num_features,num_classes=num_classes)
            num_epochs = args.epoch

        Model = Model.cuda()
        Model.train()

        optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr,weight_decay=args.decay)

        images = torch.from_numpy(X_train).float().cuda()
        label = torch.from_numpy(Y_train).long().cuda()   # gt B,H,W
        criterion = CrossEntropy2d().cuda()

        # train the classification model
        # Train time #
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            adjust_learning_rate(optimizer, args.lr, epoch, args.epoch)
            tem_time = time.time()

            optimizer.zero_grad()

            output = Model(images, label)  
            assert torch.isnan(output[0]).sum() == 0, print(output[0])

            seg_loss = criterion(output[0], output[3]) + criterion(output[1], output[4]) + criterion(output[2], output[5]) + criterion(output[0], label)

            assert torch.isnan(seg_loss).sum() == 0, print(seg_loss)
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=Model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            batch_time = time.time() - tem_time
            if (epoch+1) % 10 == 0:
                tqdm.write(
                    'epoch %d/%d:  time: %.2f cls_loss = %.4f' % (epoch + 1, num_epochs, batch_time, seg_loss.item()))
        end_time_train = time.time()
        train_times.append(end_time_train - start_time_train)  # train time end

        model_save_path = os.path.join(save_path_prefix, f"{args.model}_run{run + 1}.pth")
        torch.save(Model.state_dict(), model_save_path)
        print(f"Model checkpoint saved to {model_save_path}")

        # test the model
        Model.eval()
        with torch.no_grad():
        # ---------test---------
            start_time_test = time.time()
            output = Model(images, label)
            _, predict_labels = torch.max(output[2], 1)  # 最大值，分类预测的标签（索引）
            predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
            end_time_test = time.time()
            test_times.append(end_time_test - start_time_test)  # test time end

            end_time = time.time()
            times.append(end_time - start_time)   # run time end

            run_train_time = end_time_train - start_time_train
            run_test_time = end_time_test - start_time_test
            run_total_time = end_time - start_time
            print(f"\n[Run {run + 1}] Training Time: {run_train_time:.2f}s | Testing Time: {run_test_time:.2f}s | Total Time: {run_total_time:.2f}s")

            # computing metrics
            run_results = metrics(predict_labels[test_array], Y[test_array], n_classes=num_classes)  # only for test set
            results.append(run_results)
            show_results(run_results, label_values=label_value)

            OA = run_results["Accuracy"]
            kappa = run_results["Kappa"]

            img = DrawResult(np.reshape(predict_labels + 1, -1), args.dataID)
            plt.imsave(save_path_prefix + args.model + '_OA' + repr(int(OA * 10000)) + '_kappa' + repr(
                int(kappa * 10000)) + '.png', img)  # save result image

    if args.num_run > 1:
        avg_time_train = np.mean(train_times)
        std_time_train = np.std(train_times)

        avg_time_test = np.mean(test_times)
        std_time_test = np.std(test_times)

        avg_time = np.mean(times)
        std_time = np.std(times)

        text=show_results(results, label_values=label_value, agregated=True,file_path=log_file, avg_time_train=avg_time_train,
                          std_time_train=std_time_train, avg_time_test=avg_time_test, std_time_test=std_time_test,
                          avg_time = avg_time, std_time=std_time, opts = args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--save_exp_name', type=str, default='_15/')
    parser.add_argument('--model', type=str, default='MambaMoE')

    # train
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_run', type=int, default=10)
    parser.add_argument('--split', type=bool, default=False)
    parser.add_argument("--dataset_name", type=str, default="pu",
                        choices=['pu','whuhc','houston13']) # dataset name
    parser.add_argument("--dataset_dir", type=str, default="./Data") # dataset dir
    parser.add_argument("--train_num", type=int, default=15)

    args = parser.parse_args()
    main(args)
