import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat, reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" )

def paramsInit(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.0)
    elif isinstance(net, nn.BatchNorm2d):
        net.weight.data.fill_(1)
        net.bias.data.zero_()
    elif isinstance(net, nn.Linear):
        net.weight.data.normal_(0, 0.01)
        net.bias.data.zero_()

class CoConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(CoConvBlock, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        
        self.BN=nn.BatchNorm1d(in_dim)
        self.BN2=nn.BatchNorm1d(in_dim)
        
        self.Linear=nn.Sequential(nn.Conv1d(in_dim,out_dim,kernel_size=1)) #,nn.PReLU()
        self.Conv2D_DW=nn.Sequential(nn.Conv2d(out_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2,groups=out_dim))
        
        self.ACT=nn.PReLU()
        self.ACT2=nn.PReLU()

        self.Linear_sim = nn.Sequential(nn.Conv1d(out_dim, out_dim, kernel_size=1))
        self.score_linear = nn.Sequential(nn.Linear(out_dim, out_dim, bias=True), nn.Sigmoid())
        self.final_XH=nn.Sequential( nn.Conv1d(in_dim+out_dim, out_dim, kernel_size=1),nn.PReLU())
        paramsInit(self)

    def forward(self, X_in:torch.Tensor, H_in:torch.Tensor):
        b,c,h,w=X_in.shape
        _,_,n=H_in.shape
        
        X_in_reshape = X_in.reshape([b,c,-1])
        
        X_in_BN=self.BN(X_in_reshape)
        H_in_BN=self.BN2(H_in)
        
        # Neighbor-wise Linear Transformation
        X=self.Linear(X_in_BN)
        H=self.Linear(H_in_BN)
        
        # Temporary Vector
        X_mean=torch.mean(X,dim=-1,keepdim=True)
        X_sim=self.Linear_sim(X_mean)
        H_sim=self.Linear_sim(H)
        H_sim2=torch.cat([X_sim,H_sim],dim=-1)
        X_sim=rearrange(X_sim,'b c n -> b n c')
        similarity = torch.sigmoid( torch.matmul(X_sim,H_sim2) )
        similarity = torch.softmax(similarity, dim=-1) #b*81*5
        # similarity =similarity/(torch.sum(similarity, dim=-1,keepdim=True)+1e-15) #b*81*5
        
        # Neighbor Aggregation
        X_aggre = torch.matmul(similarity,rearrange(torch.cat([X_mean,H],dim=-1),'b c n -> b n c'))

        # Feature Fusion
        score = self.score_linear(X_aggre).permute([0,2,1]).unsqueeze(-1)
        X = self.Conv2D_DW(X.reshape([b, -1, h, w]))
        X=score* X_aggre.reshape([b, -1, 1, 1]) +(1-score)*X
        
        X=self.ACT(X)
        H=self.ACT2(H)

        # Feature Concatenation and Linear Transformation
        H_out=self.final_XH(torch.cat([H,H_in_BN],dim=1))
        X_out=self.final_XH(torch.cat([X.reshape([b,-1,h*w]),X_in_BN],dim=1)).reshape([b,-1,h,w])
        
        return X_out, H_out

class CNCMN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,learning_rate:float=0.001,classifier:str="softmax",netDepth:int=5):
        super(CNCMN, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = changel# 网络输入数据大小
        self.height = height
        self.width = width
        self.protoLen=64 # 胶囊向量长度
        self.learning_rate = learning_rate
        self.classifier=classifier
        self.netDepth=netDepth
        
        # overall prototypes
        Prototypes=torch.zeros([class_count,self.protoLen],dtype=torch.float,device=device,requires_grad=True)
        self.register_buffer('Prototypes', Prototypes)
        
        self.softmaxClassifier = nn.Sequential(nn.Linear(self.protoLen, self.class_count))
        
        self.mixconvList=nn.Sequential(CoConvBlock(changel,64,3),
                                       CoConvBlock(64,64,3),
                                       CoConvBlock(64,64,3),
                                       CoConvBlock(64,64,3),
                                       CoConvBlock(64,64,3))
        
        self.embedding=nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64,self.protoLen,kernel_size=3),
                                    nn.Tanh(), 
                                    nn.AdaptiveAvgPool2d(1))

        paramsInit(self)
        
    def unitize(self,embeddings):
        squared_norm = (embeddings ** 2).sum(dim=-1, keepdim=True)
        E_hat = embeddings / (torch.sqrt(squared_norm + 1e-15))
        return E_hat
    
    def updataPrototypes(self, embeddings, labels, ):
        '''
        update prototypes
        :param embeddings:
        :return:
        '''
        # step 1: unitize embeddings
        # E_hat=self.unitize(embeddings)
        E_hat=embeddings
        
        # step 2: similarity
        Prototypes = self.Prototypes.detach()
        S = torch.matmul(E_hat, Prototypes.t())
    
        # step 3: update prototypes
        A=(torch.clamp(1-S,0,2)*labels).t()
        D_hat = torch.diag(1. / (A.sum(1) + 1e-15))  
        batchCentroids = torch.matmul(torch.matmul(D_hat, A), E_hat) 
        # self.batchCentroids = batchCentroids = self.unitize(batchCentroids)
        self.Prototypes = self.unitize(self.learning_rate* 10 * batchCentroids + Prototypes)

        # # loss
        predict = self.metricClassifier(embeddings)
        loss = self.metricLoss(predict, labels)
        
        return predict, loss
        
    def metricLoss(self,predict,labels):
        inter_similarity = F.relu(torch.matmul(self.Prototypes, self.Prototypes.t()))  # .detach()
        left = F.relu(1 - predict, inplace=True) ** 2
        right = F.relu(predict , inplace=True) ** 2
        margin_loss = left * labels + right * (1. - labels) *0.5 #/(self.class_count-1)
        inter_weights = torch.sum(inter_similarity, dim=0, keepdim=True)
        loss = (margin_loss * inter_weights).mean()
        return loss
        
    def metricClassifier(self, embeddings):
        S = torch.matmul(embeddings, self.Prototypes.t())
        return S
    
    def softmax_loss(self,embeddings,labels):
        predict= self.softmaxClassifier(embeddings)
        loss=F.cross_entropy(predict,torch.argmax(labels,dim=-1))
        return predict, loss
    
    def getCapsules(self):
        return self.Prototypes
    
    def forward(self, data: torch.Tensor, labels:torch.Tensor, subgraphs:torch.Tensor):
        '''
        :param x: B*HW*C
        :return: probability_map H*W*C
        '''
        X=rearrange(data,'b h w c -> b c h w')
        H=rearrange(subgraphs,'b n c -> b c n')
        
        ## Feature Extraction
        for i in range(self.netDepth): X,H=self.mixconvList[i](X,H)
        
        ## Metric Space Embedding
        embeddings = self.embedding(X).squeeze(-1).squeeze(-1)

        ## prediction and loss
        pre_and_loss=[]
        if self.classifier=="softmax":
            softmaxLoss = 0
            if labels is not None:
                softmaxPredict, softmaxLoss = self.softmax_loss(embeddings, labels)
            else:
                softmaxPredict = self.softmaxClassifier(embeddings)
            pre_and_loss = [softmaxPredict, softmaxLoss]
        elif self.classifier=="metric":
            metricLoss = 0
            embeddings=self.unitize(embeddings)
            if labels is not None and self.training:
                metricPredict, metricLoss=self.updataPrototypes(embeddings,labels)
            elif labels is not None:
                metricPredict=self.metricClassifier(embeddings)
                metricLoss = self.metricLoss(metricPredict,labels)
            else:
                metricPredict = self.metricClassifier(embeddings)
            pre_and_loss = [metricPredict, metricLoss]
            
        return pre_and_loss,embeddings
        




if __name__=='__main__':
    SC=CNCMN(10,10,16)
    