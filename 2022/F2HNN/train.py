from model import HGNN_weight
from data_prepare import data_prepare_whole
import torch
import numpy as np
import random
from sklearn import metrics



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(5)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #num_class: India: 16; KSC: 13; Bot: 14; HustonU: 15
    n_class = 16
    #load data
    img_whole, whole_gt, H, W, invDE_HT, mask_TR, mask_TE, img_idx, h, w = data_prepare_whole(num_class=n_class, variable_weight=True)
    img_whole = torch.Tensor(img_whole).to(device)
    whole_gt = torch.Tensor(whole_gt).to(device)
    #DV2_H = torch.Tensor(H).to(device)
    H = torch.Tensor(H).to(device)
    W = torch.Tensor(W).to(device)
    invDE_HT = torch.Tensor(invDE_HT).to(device)
    mask_TR = torch.Tensor(mask_TR).to(device)
    mask_TE = torch.Tensor(mask_TE).to(device)


    model = HGNN_weight(
                in_ch=img_whole.shape[1],
                n_class=n_class,
                n_hid=128,
                W=W,
                dropout=0)
    model.to(device)

    num_epochs = 200
    lr = 0.01
    weight_decay = 0.0005
    milestones = [25, 50, 80]
    gamma = 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    costs = []
    costs_dev = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs + 1):
        epoch_loss = 0.
        epoch_acc = 0.
        model.train()
        optimizer.zero_grad()
        output = model(img_whole, H, invDE_HT)
        _, label = torch.max(whole_gt, 1)
        label_tr = label[mask_TR>0]
        label_te = label[mask_TE>0]
        output_tr = output[mask_TR>0,:]
        epochloss = criterion(output_tr, label_tr)
        epochloss.backward()
        optimizer.step()
        scheduler.step()
        #calculate accuracy per batch
        _, pre = torch.max(output_tr , 1)
        num_correct = torch.eq(pre, label_tr).sum().float().item()
        accuracy = num_correct/label_tr.shape[0]
        epoch_loss = epochloss
        epoch_acc = accuracy


        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output_test = model(img_whole, H, invDE_HT)
                output_te = output_test[mask_TE>0,:]
                epoch_loss_test = criterion(output_te, label_te)
                _, pre = torch.max(output_te, 1)
                num_correct = torch.eq(pre, label_te).sum().float().item()
                epoch_acc_test = num_correct / label_te.shape[0]
                print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_loss, epoch_loss_test, epoch_acc, epoch_acc_test))
                kappa = metrics.cohen_kappa_score(pre.cpu(), label_te.cpu())
                print("kappa:", kappa)










