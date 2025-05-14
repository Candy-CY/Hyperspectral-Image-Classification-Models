import torch
import torch.nn as nn
from torch.nn import Softmax
from cc_attention import CrissCrossAttention
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
# BatchNorm2d = SyncBN#functools.partial(InPlaceABNSync, activation='identity')




class SSCDNonLModel(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(SSCDNonLModel, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.nlcon2=CrissCrossAttention(CCChannel)
        self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)
        nl2=self.nlcon2(xx)
        nl2=self.nlcon2(nl2)
        nl3=self.nlcon3(xx)
        nl3=self.nlcon3(nl3)
        nl2=(nl2+nl3)*0.7+xx
        out4=torch.cat((xx, nl2),1)
        out4=self.bcat(out4)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6








