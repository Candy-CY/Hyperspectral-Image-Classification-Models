import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils.convlstm import ConvLSTM
#from torchsummaryX import summary

class specMN_scheme1(nn.Module):
    def __init__(self, spec_band, num_classes,oly_se=True, init_weights=True):
        super(specMN_scheme1, self).__init__()

        self.oly_se=oly_se
        self.spec_band = spec_band

        self.lstm1 = nn.LSTM(input_size=int(spec_band / 8), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=int(spec_band / 4), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=int(spec_band / 2), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=int(spec_band / 1), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)

        self.FC = nn.Linear(in_features=128, out_features=128)
        #self.BN = nn.BatchNorm1d(128)

        self.pre=nn.Linear(in_features=128,out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec):
        # channel of spec
        d=x_spec.shape[-1]
        # prepare data
        p1_length=int(d/8)
        p2_length=int(d/4)
        p3_length=int(d/2)
        x1=torch.zeros(x_spec.shape[0],8,p1_length).cuda()
        x2=torch.zeros(x_spec.shape[0],4,p2_length).cuda()
        x3=torch.zeros(x_spec.shape[0],2,p3_length).cuda()
        x4=x_spec.reshape(x_spec.shape[0],1,x_spec.shape[-1])
        ## eight parts
        start=0
        end = min(start + p1_length, d)
        for i in range(8):
            x1[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p1_length, d)
        ## four parts
        start = 0
        end = min(start + p2_length, d)
        for i in range(4):
            x2[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p2_length, d)
        ## two parts
        start = 0
        end = min(start + p3_length, d)
        for i in range(2):
            x3[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p3_length, d)

        _, (y_1, _) = self.lstm1(x1)#(1,B,C)
        _, (y_2, _) = self.lstm2(x2)
        _, (y_3, _) = self.lstm3(x3)
        _, (y_4, _) = self.lstm4(x4)

        y_1 = y_1.squeeze(0)
        y_2 = y_2.squeeze(0)
        y_3 = y_3.squeeze(0)
        y_4 = y_4.squeeze(0)

        #y=F.relu(self.FC(torch.cat((y_1,y_2,y_3,y_4),1)))
        y=y_1+y_2+y_3+y_4

        y = F.relu(self.FC(y))

        if self.oly_se:
            score=self.pre(y)
            return score

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class specMN_scheme2(nn.Module):

    def __init__(self, spec_band, num_classes, strategy='s1',time_steps=3,oly_se=True, init_weights=True):
        super(specMN_scheme2, self).__init__()

        self.oly_se=oly_se
        self.strategy=strategy
        self.time_steps=time_steps
        self.spec_band = spec_band

        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2,stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2,stride=2)
        )

        p0_length = spec_band
        p1_length = p0_length//2
        p2_length = p1_length//2
        p3_length = p2_length//2

        if strategy=='s1':

            self.lstm1 = nn.LSTM(input_size=int(p1_length / 8), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / 4), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / 2), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)

        if strategy=='s2':

            self.lstm1 = nn.LSTM(input_size=int(p1_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)

        self.FC=nn.Linear(in_features=128,out_features=128)
        #self.BN=nn.BatchNorm1d()

        self.pre = nn.Linear(in_features=128, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec):
        # change the shape
        x_spec=x_spec.reshape(-1,1,x_spec.shape[-1])

        x1=self.conv2(x_spec)
        x2=self.conv3(x1)
        x3=self.conv4(x2)

        x1=x1.squeeze(1)
        x2=x2.squeeze(1)
        x3=x3.squeeze(1)

        if self.strategy=='s1':
             # prepare data
            p1_length=int(x1.shape[-1]/8)
            p2_length=int(x2.shape[-1]/4)
            p3_length=int(x3.shape[-1]/2)
            x1_r=torch.zeros(x_spec.shape[0],8,p1_length)
            x2_r=torch.zeros(x_spec.shape[0],4,p2_length)
            x3_r=torch.zeros(x_spec.shape[0],2,p3_length)
            ## eight parts
            start=0
            end = min(start + p1_length, x1.shape[-1])
            for i in range(8):
                x1_r[:, i, :] = x1[:,start:end]
                start = end
                end = min(start + p1_length, x1.shape[-1])
            ## four parts
            start = 0
            end = min(start + p2_length, x2.shape[-1])
            for i in range(4):
                x2_r[:, i, :] = x2[:, start:end]
                start = end
                end = min(start + p2_length, x2.shape[-1])
            ## two parts
            start = 0
            end = min(start + p3_length, x3.shape[-1])
            for i in range(2):
                x3_r[:, i, :] = x3[:, start:end]
                start = end
                end = min(start + p3_length, x3.shape[-1])

        if self.strategy=='s2':
            # prepare data
            p1_length = int(x1.shape[-1] / self.time_steps)
            p2_length = int(x2.shape[-1] / self.time_steps)
            p3_length = int(x3.shape[-1] / self.time_steps)
            x1_r = torch.zeros(x_spec.shape[0], self.time_steps, p1_length)
            x2_r = torch.zeros(x_spec.shape[0], self.time_steps, p2_length)
            x3_r = torch.zeros(x_spec.shape[0], self.time_steps, p3_length)
            ## time steps parts
            start1 = 0
            end1 = min(start1 + p1_length, x1.shape[-1])
            start2 = 0
            end2 = min(start2 + p2_length, x2.shape[-1])
            start3 = 0
            end3 = min(start3 + p3_length, x3.shape[-1])
            for i in range(self.time_steps):
                x1_r[:, i, :] = x1[:, start1:end1]
                x2_r[:, i, :] = x2[:, start2:end2]
                x3_r[:, i, :] = x3[:, start3:end3]
                start1 = end1
                end1 = min(start1 + p1_length, x1.shape[-1])
                start2 = end2
                end2 = min(start2 + p2_length, x2.shape[-1])
                start3 = end3
                end3 = min(start3 + p3_length, x3.shape[-1])

        x1_r,x2_r,x3_r=x1_r.cuda(),x2_r.cuda(),x3_r.cuda()

        _, (y_1, _) = self.lstm1(x1_r)
        _, (y_2, _) = self.lstm2(x2_r)
        _, (y_3, _) = self.lstm3(x3_r)
        _, (y_4, _) = self.lstm4(x_spec)

        y_1 = y_1.squeeze(0)
        y_2 = y_2.squeeze(0)
        y_3 = y_3.squeeze(0)
        y_4 = y_4.squeeze(0)

        y = y_1 + y_2 + y_3 + y_4

        y=F.relu(self.FC(y))

        if self.oly_se:
            score=self.pre(y)
            return score

        return y

    # ref https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class spatMN(nn.Module):
    def __init__(self,wsz,strd,spat_channel,num_classes,num_patch_inrow,drop_p,group,seq,oly_sa=True,init_weights=True):
        super(spatMN, self).__init__()

        self.strd=strd
        self.wsz=wsz#split size
        self.oly_sa=oly_sa
        self.npi=num_patch_inrow
        self.group=group
        self.seq=seq

        self.CL_1 = nn.ModuleList()
        self.CL_2 = nn.ModuleList()
        self.CL_3 = nn.ModuleList()
        self.CONV = nn.ModuleList()
        self.FC = nn.ModuleList()

        y_length=0
        isz=27#input size
        #input_dim=32

        for i in range(3):
            input_dim = spat_channel if i == 0 else 32
            #if i<2:
            self.CL_tmp = nn.ModuleList()
            self.CV_tmp = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=(3, 3),
                          stride=1, padding=1, dilation=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3, 3),
                         stride=1, padding=2, dilation=2, bias=True),
                nn.ReLU()
            )
            for j in range(num_patch_inrow):
                self.CL_tmp.append(ConvLSTM(input_size=(wsz[i], wsz[i]), input_dim=32,
                                          hidden_dim=32, kernel_size=(3, 3), num_layers=1,
                                          batch_first=True, bias=True, return_all_layers=False))
            self.CL_2.append(ConvLSTM(input_size=(wsz[i], wsz[i]), input_dim=32,
                                      hidden_dim=32, kernel_size=(3, 3), num_layers=1,
                                      batch_first=True, bias=True, return_all_layers=False))
            self.CL_1.append(self.CL_tmp)


            self.CONV.append(self.CV_tmp)

            isz=27//self.npi
            y_length=isz*isz*32
            #y_length=27*27*32#no CL
            self.FC.append(nn.Linear(in_features=y_length,out_features=128))

        self.CL = nn.ModuleList([self.CL_1,self.CL_2])
        self.FC_=nn.Linear(in_features=128, out_features=128)
        self.pre = nn.Linear(in_features=128, out_features=num_classes)

        self.drop=nn.Dropout(p=drop_p)

        if init_weights:
            self._initialize_weights()

    def forward(self,x_spat):
        B,_,H,W=x_spat.size()
        x=x_spat
        merge=[]
        for i in range(3):
            #if i < 2:
            x = self.CONV[i](x)

            t1=self.npi
            t2=self.npi
            input = torch.zeros(B, t1*t2, x.shape[1], self.wsz[i], self.wsz[i]).cuda() # (b,t,c,h,w)
            count=0

            stone = x.shape[-1] // self.npi * self.npi

            # traditional strategy
            if self.group == 'traditional':
                for j in range(0,stone,self.strd[i]):
                    for k in range(0,stone,self.strd[i]):
                        if j+self.wsz[i]<=stone and k+self.wsz[i]<=stone:
                            input[:,count,:,:,:]=x[:,:,j:j+self.wsz[i],k:k+self.wsz[i]]
                            count+=1
            # alternate strategy
            elif self.group == 'alternate':
                for j in range(0,t1):
                    for k in range(0,t2):
                        input[:,count,:,:,:]=x[:,:,j:stone:self.npi,k:stone:self.npi]
                        count+=1
            else:
                raise NotImplementedError

            # plain style
            if self.seq=='plain':
                _, output_ = self.CL[-1][i](input)
            # cascade style
            elif self.seq=='cascade':
                start=0
                end=self.npi
                output=[]
                for ii in range(self.npi):
                    input_tmp=input[:,start:end,:,:,:]
                    #print(ii,i)
                    _, output_tmp = self.CL[0][i][ii](input_tmp)
                    output.append(output_tmp[0][0])
                    start=end
                    end=np.min([end+self.npi,t1*t2])

                input_=torch.stack(output,1)
                _,output_= self.CL[-1][i](input_)
            else:
                raise NotImplementedError

            merge.append(output_[0][0])
            #merge.append(x)#no CL

        merge=list(map(lambda x:x.reshape(B,-1),merge))

        for i in range(3):
            if i==0:
                y=self.FC[i](self.drop(merge[i]))
                #y=self.FC[i](merge[i])
            else:
                y+=self.FC[i](self.drop(merge[i]))
                #y+= self.FC[i](merge[i])

        y = self.FC_(y)

        y=F.relu(y)

        if self.oly_sa:
            score=self.pre(y)
            return score
        return y

    # ref https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class ASSMN(nn.Module):
    def __init__(self,spec_band,spat_channel,scheme,strategy,spec_time_steps,wsz,strd,
                 num_classes,num_patch_inrow,drop_p,group,seq,oly_se,oly_sa,init_weights=True):
        super(ASSMN,self).__init__()

        self.oly_se=oly_se
        self.oly_sa=oly_sa
        self.scheme=scheme
        self.se1=specMN_scheme1(spec_band, num_classes,oly_se=oly_se)
        self.se2=specMN_scheme2(spec_band, num_classes, strategy=strategy,time_steps=spec_time_steps,oly_se=oly_se)
        self.sa=spatMN(wsz,strd,spat_channel,num_classes,num_patch_inrow,drop_p,group,seq,oly_sa=oly_sa)

        self.gamma=nn.Parameter(torch.randn(1))
        self.pre = nn.Linear(in_features=128, out_features=num_classes)
        self.pre1 = nn.Linear(in_features=128, out_features=num_classes)
        self.pre2 = nn.Linear(in_features=128, out_features=num_classes)
        self.pre_concat = nn.Linear(in_features=256, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec,x_spat):

        if self.oly_se and not self.oly_sa:
            if self.scheme==1:
                se=self.se1(x_spec)
            if self.scheme==2:
                se=self.se2(x_spec)
            return [se]

        elif self.oly_sa and not self.oly_se:
            sa=self.sa(x_spat)
            return [sa]

        elif not self.oly_se and not self.oly_sa:
            if self.scheme==1:
                se=self.se1(x_spec)
            if self.scheme==2:
                se=self.se2(x_spec)
            sa = self.sa(x_spat)
            gamma=torch.sigmoid(self.gamma)
            #x=gamma*se+(1-gamma)*sa
            #score=self.pre(x)
            #score=self.pre_concat(torch.cat((se,sa),1))
            score_se1 = self.pre1(se)
            score_sa1 = self.pre2(sa)
            # score_se2 = self.pre3(se)
            # score_sa2 = self.pre4(sa)
            score=gamma*score_se1+(1-gamma)*score_sa1
            return [score,score_se1,score_sa1]
        else:
            raise NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                nn.init.constant_(m.bias.data, 0)

class operate():

    def train(self, epoch, loss_trn, net, optimizer, scheduler, trn_loader, criterion):
        net.train()  # train mode
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spec, X_spat, y_target) in enumerate(trn_loader):
            X_spec, X_spat=Variable(X_spec.float()).cuda(), Variable(X_spat.float()).cuda()
            ######GPU
            y_target = Variable(y_target.float().long()).cuda()
            y_pred = net.forward(X_spec, X_spat)
            for i in range(len(y_pred)):
                if i==0:
                    loss = criterion(y_pred[i], y_target)
                if i>0:
                    loss+=criterion(y_pred[i], y_target)

            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred[0].data, 1)
            # print(torch.unique(predicted))
            # print(torch.unique(y_target))
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if idx % 20==0:

            del X_spec, X_spat, y_target
            del y_pred
            # del loss
        scheduler.step()
        print('train epoch:{},train loss:{},correct/total:{:.4f}%'.format(epoch,
               epochavg_loss / (idx + 1),100 * correct.item() / total))
        loss_trn.append(epochavg_loss / (idx + 1))
        return loss_trn

    def inference(self,net, data_loader, criterion, FLAG='VAL'):
        net.eval()  # evaluation mode
        inf_loss = 0
        num = 1
        correct = 0
        total = 0
        for idx, (X_spec, X_spat, y_target) in enumerate(data_loader):
            with torch.no_grad():
                X_spec, X_spat=\
                    Variable(X_spec.float()).cuda(), Variable(X_spat.float()).cuda()
            ######GPU
                y_target = Variable(y_target.float().long()).cuda()
                y_score = net.forward(X_spec, X_spat)

            for i in range(len(y_score)):
                if i==0:
                    loss = criterion(y_score[i], y_target)
                if i>0:
                    loss+=criterion(y_score[i], y_target)

            inf_loss += loss.float()

            _, predicted = torch.max(y_score[0].data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            y_pred_inf = np.argmax(y_score[0].detach().cpu().numpy(), axis=1) + 1
            if num == 1:
                inf_result = y_pred_inf
            else:
                inf_result = np.hstack((inf_result, y_pred_inf))
            if idx % 20 == 0 and idx > 0:
                print('test loss:{},{}/{}({:.2f}%),correct/total:{:.4f}%'.format(
                    loss.item(), idx * X_spec.shape[0],len(data_loader.dataset),100 * idx * X_spec.shape[0] / len(
                    data_loader.dataset),100 * correct.item() / total))
            num += 1
            del X_spec,X_spat, y_target
            del loss
            del y_score
            del y_pred_inf
        avg_inf_loss = inf_loss / len(data_loader.dataset)
        if FLAG == 'VAL':
            print('Over all validation loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        elif FLAG == 'TEST':
            print('Over all testing loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        return inf_result


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model=ASSMN(200,4,2,'s2',2,[9,9,9],[9,9,9],16,3,0.8,'alternate','cascade',False,False,init_weights=True)
    model.cuda()
   #summary(model, torch.zeros((1, 200)).cuda(),torch.zeros((1,4,27,27)).cuda())