import numpy as np
import time
import torch
from torch.autograd import Variable


per_out = []
per_out_Y = []

class operate():
    def train(self, epoch, loss_trn, acc_trn, tsne_trn, tsne_trn_y, net, optimizer, scheduler, trn_loader, criterion, ):
        start = time.time()
        net.train()
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spe, X_spat, y_target) in enumerate(trn_loader):
            X_spe = Variable(X_spe.float()).cuda()
            X_spat = Variable(X_spat.float()).cuda()
            y_target = Variable(y_target.float().long()).cuda()
            output = net.forward(X_spe, X_spat)
            y_pred = output[0]

            loss = 0.
            for j in range(3):
                if j == 0:
                    loss_main = criterion(output[j], y_target)
                    loss = loss_main
                elif j == 1:
                    loss_Spe = criterion(output[j], y_target)
                    loss += loss_Spe / (loss_Spe / loss_main).detach()
                elif j == 2:
                    loss_Spa = criterion(output[j], y_target)
                    loss += loss_Spa / (loss_Spa / loss_main).detach()


            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del X_spat
            del y_pred

        scheduler.step()
        end = time.time()
        print('train epoch:{},train loss:{},correct/total:{:.4f}%,time:{:.1f}s'.format(epoch,
                                                                                       epochavg_loss / (idx + 1),
                                                                                       100 * correct.item() / total,
                                                                                       end - start))
        loss_trn.append(epochavg_loss / (idx + 1))
        acc_trn.append(correct.item() / total)

        return loss_trn, acc_trn, tsne_trn, tsne_trn_y

    def inference(self, tsne_tes, tsne_tes_y, net, data_loader, criterion, FLAG='VAL'):
        net.eval()  # evaluation mode
        inf_loss = 0
        num = 1
        correct = 0
        total = 0
        for idx, (X_spe, X_spat, y_target) in enumerate(data_loader):
            with torch.no_grad():
                X_spe = Variable(X_spe.float()).cuda()
                X_spat = Variable(X_spat.float()).cuda()
                y_target = Variable(y_target.float().long()).cuda()
                output = net.forward(X_spe, X_spat)
            y_score = output[0]

            loss = 0.
            for j in range(3):
                if j == 0:
                    loss += 1.0 * criterion(output[j], y_target) 
                elif j == 1:
                    loss += 0.3 * criterion(output[j], y_target) 
                elif j == 2:
                    loss += 0.9 * criterion(output[j], y_target)  


            inf_loss += loss.float()

            _, predicted = torch.max(y_score.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            y_pred_inf = np.argmax(y_score.detach().cpu().numpy(), axis=1) + 1
            if num == 1:
                inf_result = y_pred_inf
            else:
                inf_result = np.hstack((inf_result, y_pred_inf))
            num += 1

            del X_spe, X_spat, y_target
            del loss
            del y_score
            del y_pred_inf
        avg_inf_loss = inf_loss / len(data_loader.dataset)

        return inf_result, tsne_tes, tsne_tes_y
