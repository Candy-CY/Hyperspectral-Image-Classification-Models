import numpy as np
import torch
import torch.utils.data as data

import os
import matplotlib.pyplot as plt
import matplotlib.image as mi
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from torch.optim import AdamW

from Models import DualViT, HSIViT
from Utils.Preprocessing import get_data_set_dual, spilt_dataset
from Utils.Label_to_Colormap import label_to_colormap
from Utils.Seed_Everything import seed_everything, stable

from timm.scheduler import CosineLRScheduler

from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')


class HSIdataset(data.Dataset):
    def __init__(self, data_list, gt=None, train=False, device='cuda:0'):
        self.data_list = data_list
        self.gt = gt

        self.train = train
        self.device = device

    def random_horizontal_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 1)
        else:
            return data

    def random_vertical_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 0)
        else:
            return data

    def __getitem__(self, index):
        data = self.data_list[index]
        if self.train:
            data = self.random_horizontal_filp(data)
            data = self.random_vertical_filp(data)

        data = torch.tensor(data.copy(), dtype=torch.float32)
        data = data.unsqueeze(0).permute(0, 3, 1, 2)
        if self.gt is not None:
            gt = self.gt[index]
            return data, gt
        else:
            return data

    def __len__(self):
        return len(self.data_list)


def dual_branch_finetuning(data_list, labeled_index, unlabeled_data, gt, save_dir, model_name, pretrained=None,
                           lr=1e-3, wd=5e-3, depth=12, dim=144, dec_depth=2, dec_dim=72, s_depth=6,
                           epochs=100, mask_ratio=0.5, lamda=5, batch_size=32):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h, w, c = data_list[0].shape

    n_class = np.max(gt) + 1
    print('number of class: ', n_class)

    model = DualViT(img_size=h, patch_size=3, in_chans=1, bands=c, b_patch_size=8, num_class=n_class,
                    embed_dim=dim, depth=depth, num_heads=dim // 16, s_depth=s_depth,
                    decoder_embed_dim=dec_dim, decoder_depth=dec_depth, decoder_num_heads=dec_dim // 8,
                    norm_pix_loss=True, trunc_init=True, drop_path=0.2).to(device)

    save_path = os.path.join(save_dir, model_name.replace('.pkl', ''))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pre_keys = []
    ignore_keys = []
    if pretrained:
        state_dict = {}
        model_dict = model.state_dict()
        pretrain_model_para = torch.load(pretrained, map_location=device)
        for key, v in pretrain_model_para.items():
            if (key in model_dict.keys()) & (key not in ignore_keys):
                state_dict[key] = v
                pre_keys.append(key)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    no_decay = ['bias', 'norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd)

    warm_up_epoch = int(np.ceil(0.1 * epochs))
    scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=lr * 0.01, warmup_t=warm_up_epoch,
                                  warmup_lr_init=lr * 0.01)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    train_data_index, train_gt, val_data_index, val_gt = spilt_dataset(labeled_index, gt, training_ratio=0.5)

    train_labeled_dataset = HSIdataset(data_list[train_data_index], train_gt, train=True)
    unlabeled_dataset = HSIdataset(unlabeled_data, train=True)
    val_labeled_dataset = HSIdataset(data_list[val_data_index], val_gt)
    print('dataset load finished')
    print('训练集大小：' + str(len(train_labeled_dataset)))

    train_labeled_dataload = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_batch_size = int(np.ceil(len(unlabeled_dataset) / len(train_labeled_dataload)) / 2)
    unlabeled_dataload = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True)
    val_labeled_dataload = DataLoader(val_labeled_dataset, batch_size=512, shuffle=False)
    print('batch load finished')
    print('训练轮次：' + str(len(train_labeled_dataload)))

    epoch_loss_list = []
    val_loss_list = []
    epoch_AA_list = []
    val_AA_list = []
    iter_num = 0

    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avarage Accuracy')

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        pred = np.zeros(1)
        gt_ = np.zeros(1)
        labeled_iter = iter(stable(train_labeled_dataload, 42 + epoch))
        unlabeled_iter = iter(stable(unlabeled_dataload, 42 + epoch))

        for batch_idx in range(len(train_labeled_dataload)):
            x, y = labeled_iter.next()
            x_u = unlabeled_iter.next()
            loss_rec, _, _, outputs = model(x.to(device), x_u.to(device), mask_ratio=mask_ratio)

            targets = y.long().to(device)
            loss_cls = criterion(outputs, targets)
            loss = lamda * loss_rec + loss_cls

            outputs = outputs.detach().cpu().numpy()
            output = np.argmax(outputs, axis=1)
            pred = np.concatenate([pred, output], axis=0)

            gt = y.numpy()
            gt_ = np.concatenate([gt_, gt], axis=0)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            iter_num += 1
            train_loss += loss.item()

        pred = pred[1:]
        gt_ = gt_[1:]
        gt_label = gt_[gt_ != 0] - 1
        pred_label = pred[gt_ != 0] - 1

        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        train_aa = (oa + aa + kappa) / 3
        epoch_AA_list.append(train_aa)

        tloss = train_loss / len(train_labeled_dataload)
        epoch_loss_list.append(tloss)

        model.eval()
        pred = np.zeros(1)
        gt_ = np.zeros(1)
        with torch.no_grad():
            val_loss = 0
            labeled_iter = iter(stable(val_labeled_dataload, 42 + epoch))
            for batch_idx in range(len(val_labeled_dataload)):
                x, y = labeled_iter.next()
                outputs = model(x.to(device), mask_ratio=mask_ratio)
                targets = y.long().to(device)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                outputs = outputs.detach().cpu().numpy()
                output = np.argmax(outputs, axis=1)
                pred = np.concatenate([pred, output], axis=0)

                gt = y.numpy()
                gt_ = np.concatenate([gt_, gt], axis=0)

        pred = pred[1:]
        gt_ = gt_[1:]
        gt_label = gt_[gt_ != 0] - 1
        pred_label = pred[gt_ != 0] - 1

        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        ca = metrics.recall_score(gt_label, pred_label, average=None)
        val_value = [oa, aa, kappa, ca]

        val_aa = (oa + aa + kappa) / 3
        vloss = val_loss / len(val_labeled_dataload)

        val_AA_list.append(val_aa)
        val_loss_list.append(vloss)

        if (epoch + 1) == epochs:
            ax1.cla()
            ax2.cla()
            ln1 = ax1.plot(epoch_loss_list, 'b', lw=1, label='train loss')
            ln2 = ax1.plot(val_loss_list, 'g', lw=1, label='val loss')
            ln3 = ax2.plot(epoch_AA_list, 'y', lw=1, label='train aa')
            ln4 = ax2.plot(val_AA_list, 'r', lw=1, label='val aa')
            lns = ln1 + ln2 + ln3 + ln4
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc='center right')
            plt.pause(0.1)
        scheduler.step(epoch)

    torch.save(model.state_dict(), os.path.join(save_dir, model_name))

    plt.savefig(save_path + '/finetune_loss_' + str(lr) + '.png')
    plt.close()
    return val_value, epoch_loss_list, val_loss_list


def test_model(data_cubes, test_gt, gt, save_dir, model_name, depth=12, dim=96, s_depth=6):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    h, w, c = data_cubes[0].shape
    n_class = np.max(gt) + 1

    model = HSIViT(img_size=h, patch_size=3, in_chans=1, bands=c, b_patch_size=8,
                   num_class=n_class, embed_dim=dim, depth=depth, num_heads=dim // 16, s_depth=s_depth,
                   sep_pos_embed=True, use_learnable_pos_emb=False).to(device)

    state_dict = {}
    model_dict = model.state_dict()
    pretrain_model_para = torch.load(os.path.join(save_dir, model_name), map_location=device)
    for key, v in pretrain_model_para.items():
        if key in model_dict.keys():
            state_dict[key] = v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    dataset = HSIdataset(data_cubes)
    test_dataload = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    save_path = os.path.join(save_dir, model_name.replace('.pkl', ''))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred = np.zeros(1)
    with torch.no_grad():
        for x in tqdm(test_dataload):
            inputs = x.to(device)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            output = np.argmax(outputs[:, 1:], axis=1)
            pred = np.concatenate([pred, output], axis=0)

        pred = pred[1:] + 1
        pred = pred.reshape(gt.shape)
        colormap_all = label_to_colormap(pred)

        pred[gt == 0] = 0
        colormap = label_to_colormap(pred)

        gt_ = test_gt.reshape(-1)
        gt_label = gt_[gt_ != 0] - 1

        pred_ = pred.reshape(-1)
        pred_label = pred_[gt_ != 0] - 1

        # cm = metrics.confusion_matrix(gt_label, pred_label)
        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        ca = metrics.recall_score(gt_label, pred_label, average=None)

        mi.imsave(os.path.join(save_path, model_name.replace('.pkl', '_all_oa_' + str(np.around(oa * 100, 2)) + '.png')), colormap_all)
        mi.imsave(os.path.join(save_path, model_name.replace('.pkl', '_oa_' + str(np.around(oa * 100, 2)) + '.png')), colormap)
    return oa, aa, kappa, ca


if __name__ == "__main__":
    seeds = [3407, 3408, 3409, 3410, 3411]

    wd = 5e-3
    epoch = 200
    batch_size = 32

    enc_paras = [12, 256, 9]  # [depth, dimension, spatial-spectral encoders depth], [12, 128, 9] for HSIMAE-Base, [12, 256, 9] for Large
    dec_paras = [8, 64]  # [depth, dimension]

    mask_ratio = 0.8
    lamda = 10

    patch_size = 9
    labeled_num = 40

    model_name = 'HSIMAE_L.pkl'

    pretrained_model = "model path"
    data_path = "data path"  # numpy array, [h, w, Bands]
    gt_path = "label path"  # numpy array, [h, w]
    save_path = "save path"

    report_test_results = True

    lrs = [5e-3, 1e-3, 5e-4, 1e-4]
    best_score = []
    for lr in lrs:
        val_results = []

        for i in range(3):
            seed_everything(seeds[i])
            labeled_index, train_gt, unlabeled_data, data_set, test_gt, gt = get_data_set_dual(data_path,
                                                                                               gt_path,
                                                                                               patch_size=patch_size,
                                                                                               num=labeled_num,
                                                                                               norm=False)

            [oa, aa, kappa, ca], train_loss, val_loss = dual_branch_finetuning(data_set,
                                                                               labeled_index,
                                                                               unlabeled_data,
                                                                               train_gt,
                                                                               save_path,
                                                                               model_name,
                                                                               pretrained=pretrained_model,
                                                                               lr=lr,
                                                                               wd=wd,
                                                                               epochs=epoch,
                                                                               batch_size=batch_size,
                                                                               depth=enc_paras[0],
                                                                               dim=enc_paras[1],
                                                                               s_depth=enc_paras[2],
                                                                               dec_depth=dec_paras[0],
                                                                               dec_dim=dec_paras[1],
                                                                               mask_ratio=mask_ratio,
                                                                               lamda=lamda,
                                                                               )

            val_results.append([oa, aa, kappa])

        val_results = np.array(val_results)
        val_mean = np.mean(val_results, axis=0)
        val_std = np.std(val_results, axis=0)

        if best_score:
            if best_score[0].mean() < val_mean.mean():
                best_score = [val_mean, val_std, lr]
            else:
                best_score = best_score
        else:
            best_score = [val_mean, val_std, lr]

    if report_test_results:
        lr = best_score[2]
        test_results = []
        test_results_per_class = []
        for i in range(5):
            seed_everything(seeds[i])
            labeled_index, train_gt, unlabeled_data, data_set, test_gt, gt = get_data_set_dual(data_path,
                                                                                               gt_path,
                                                                                               patch_size=patch_size,
                                                                                               num=labeled_num,
                                                                                               norm=False)

            [oa, aa, kappa, ca], train_loss, val_loss = dual_branch_finetuning(data_set,
                                                                               labeled_index,
                                                                               unlabeled_data,
                                                                               train_gt,
                                                                               save_path,
                                                                               model_name,
                                                                               pretrained=pretrained_model,
                                                                               lr=lr,
                                                                               wd=wd,
                                                                               epochs=epoch,
                                                                               batch_size=batch_size,
                                                                               depth=enc_paras[0],
                                                                               dim=enc_paras[1],
                                                                               s_depth=enc_paras[2],
                                                                               dec_depth=dec_paras[0],
                                                                               dec_dim=dec_paras[1],
                                                                               mask_ratio=mask_ratio,
                                                                               lamda=lamda,
                                                                               )

            oa, aa, kappa, ca = test_model(data_set,
                                           test_gt,
                                           gt,
                                           save_path,
                                           model_name,
                                           depth=enc_paras[0],
                                           dim=enc_paras[1],
                                           s_depth=enc_paras[2],
                                           )

            test_results.append([oa, aa, kappa])
            test_results_per_class.append(ca)

        test_results = np.array(test_results)
        test_mean = np.mean(test_results, axis=0)
        test_std = np.std(test_results, axis=0)

        test_results_per_class = np.array(test_results_per_class)
        class_accuracy_mean = np.mean(test_results_per_class, axis=0) * 100
        class_accuracy_std = np.std(test_results_per_class, axis=0) * 100

        results = [class_accuracy_mean, test_mean, test_std, lr]
    else:
        results = [[], best_score[0], best_score[1], best_score[2]]

    print('test oa, aa, kappa of Salinas:')
    print(results[1])

    print('class_accuracy:')
    for ca in results[0]:
        print(np.around(ca, 2))

    print('best learning rate: ')
    print(results[3])

    print('test oa, aa, kappa: ')
    for mean in results[1]:
        print(np.around(mean * 100, 2))
    for var in results[2]:
        print(np.around(var * 100, 2))