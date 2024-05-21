import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable



class operate():
    def train(self, epoch, loss_trn, net, optimizer, scheduler, trn_loader, trn_loader_L, trn_loader_S, criterion, ):
        start = time.time()
        net.train()  # train mode
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spat_L, X_spat_S, y_target) in enumerate(trn_loader):

            X_spat_L = Variable(X_spat_L.float()).cuda()  # cuda(): move data to GPU
            X_spat_S = Variable(X_spat_S.float()).cuda()  # cuda(): move data to GPU
            y_target = Variable(y_target.float().long()).cuda()

            y_pred = net.forward(X_spat_L, X_spat_S)

            loss = criterion(y_pred, y_target)

            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del X_spat_L, X_spat_S, y_target
            del y_pred
        scheduler.step()
        end = time.time()
        print('train epoch:{},train loss:{},correct/total:{:.4f}%,time:{:.1f}s'.format(epoch,
                                                                                       epochavg_loss / (idx + 1),
                                                                                       100 * correct.item() / total,
                                                                                       end - start))
        loss_trn.append(epochavg_loss / (idx + 1))

        return loss_trn

    def train_CNN(self, epoch, loss_trn, net, optimizer, scheduler, trn_loader, criterion):
        start = time.time()
        net.train()  # train mode
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spat, y_target) in enumerate(trn_loader):
            X_spat = Variable(X_spat.float()).cuda()

            y_target = Variable(y_target.float().long()).cuda()
            y_pred = net.forward(X_spat, epoch)

            if epoch == 200 - 1:
                global per_out_Y
                per_out_Y.append(y_target)

            loss = criterion(y_pred, y_target)

            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)

            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del X_spat, y_target
            del y_pred

        scheduler.step()
        end = time.time()
        print('train epoch:{},train loss:{},correct/total:{:.4f}%,time:{:.1f}s'.format(epoch,
                                                                                       epochavg_loss / (idx + 1),
                                                                                       100 * correct.item() / total,
                                                                                       end - start))
        loss_trn.append(epochavg_loss / (idx + 1))

        if epoch == 200 - 1:
            return per_out_Y
        else:
            return loss_trn

    def inference(self, net, data_loader, criterion, FLAG='VAL'):
        net.eval()  # evaluation mode
        inf_loss = 0
        num = 1
        correct = 0
        total = 0
        for idx, (X_spat_L, X_spat_S, y_target) in enumerate(data_loader):
            with torch.no_grad():
                X_spat_L = Variable(X_spat_L.float()).cuda()  # GPU
                X_spat_S = Variable(X_spat_S.float()).cuda()  # GPU
                y_target = Variable(y_target.float().long()).cuda()
                y_score = net.forward(X_spat_L, X_spat_S)
            loss = criterion(y_score, y_target)
            inf_loss += loss.float()  # save memory

            _, predicted = torch.max(y_score.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            y_pred_inf = np.argmax(y_score.detach().cpu().numpy(), axis=1) + 1
            if num == 1:
                inf_result = y_pred_inf
            else:
                inf_result = np.hstack((inf_result, y_pred_inf))
            if FLAG == 'TEST' and idx % 200 == 0 and idx > 0:
                print('test loss:{},{}/{}({:.2f}%),correct/total:{:.4f}%'.format(
                    loss.item(), idx * X_spat_L.shape[0], len(data_loader.dataset), 100 * idx * X_spat_L.shape[0] / len(
                        data_loader.dataset), 100 * correct.item() / total))
            num += 1
            del X_spat_L, X_spat_S, y_target
            del loss
            del y_score
            del y_pred_inf
        avg_inf_loss = inf_loss / len(data_loader.dataset)
        if FLAG == 'VAL':
            print('Over all validation loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        if FLAG == 'TEST':
            print('Over all testing loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        return inf_result