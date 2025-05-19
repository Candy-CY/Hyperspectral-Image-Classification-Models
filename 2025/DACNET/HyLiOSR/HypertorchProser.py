import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
# import ipdb
import torch.nn.init as init

# color
colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                   [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                   [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
                   [0, 0, 128]])

class ResNet999(nn.Module):
    def __init__(self, data,band1, band2, dummynum, ncla1,n_sub_prototypes=3,latent_dim=10,l=1,temp_intra = 1,temp_inter = 0.1):
        super(ResNet999, self).__init__()
        self.data = data
        self.n_sub_prototypes = n_sub_prototypes
        self.latent_dim = latent_dim
        self.n_classes = ncla1
        self.temp_intra = temp_intra
        self.temp_inter = temp_inter
        
        if self.data == 3:
            self.conv0x = nn.Conv2d(band2+1, 32, kernel_size=(3, 3), padding='valid')
            self.conv0 = nn.Conv2d(band1-1, 32, kernel_size=(3, 3), padding='valid')
        else:
            self.conv0x = nn.Conv2d(band2, 32, kernel_size=(3, 3), padding='valid')
            self.conv0 = nn.Conv2d(band1, 32, kernel_size=(3, 3), padding='valid')
        self.bn11 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.bn21 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.fc1 = nn.Linear(64, self.n_classes)
        self.fc2 = nn.Linear(64, dummynum)

        self.fc_mu = nn.Linear(64, latent_dim) 
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.prototypes = nn.Parameter(torch.randn(self.n_classes*self.n_sub_prototypes, self.latent_dim).cuda(), requires_grad=True) 

        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 1), padding=0)
        self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)
        self.dconv3 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)
        self.dconv4 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)

        if self.data == 3:
            self.dconvhsi = nn.ConvTranspose2d(32, band1-1, kernel_size=(3, 3), padding=0)
            self.dconvlidar = nn.ConvTranspose2d(32, 2, kernel_size=(3, 3), padding=0)
        else:
            self.dconvhsi = nn.ConvTranspose2d(32, band1, kernel_size=(3, 3), padding=0)
            self.dconvlidar = nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), padding=0)

        self.bn1_de = nn.BatchNorm2d(64)
        self.bn2_de = nn.BatchNorm2d(64)

        
        init.normal_(self.conv0x.weight, mean=0.0, std=0.01)
        init.normal_(self.conv0.weight, mean=0.0, std=0.01)
        init.normal_(self.conv11.weight, mean=0.0, std=0.01)
        init.normal_(self.conv12.weight, mean=0.0, std=0.01)
        init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        init.normal_(self.fc_mu.weight, mean=0.0, std=0.01)
        init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
    
    def encoder(self, input):

        input1,input2 = torch.split(input, [input.shape[1]-2, 2], dim=1)


        x1 = self.conv0(input1)
        x1x = self.conv0x(input2)
        x1 = torch.cat([x1, x1x], dim=1)

        x11 = F.relu(self.bn11(x1))
        x11 = F.relu(self.conv11(x11))
        x11 = self.conv12(x11)
        x1 = x1 + x11

        return x1

    def latter(self, x1):

        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)
        
        return x1

    def distance(self, latent_z, prototypes): 
        matrixA, matrixB = latent_z, prototypes 
        matrixB_t = matrixB.transpose(0,1)
        matrixA_square = torch.sum(torch.pow(matrixA,2),1, keepdim=True)
        matrixB_square = torch.sum(torch.pow(matrixB_t,2),0, keepdim=True)
        product_A_B = torch.matmul(matrixA, matrixB_t)
        distance_matrix = matrixA_square + matrixB_square - 2 * product_A_B
        return distance_matrix

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        kl_div = self.distance(mean, prototypes) 
        kl_div += torch.sum((logvar.exp() - logvar - 1), axis=1, keepdims=True)
        return 0.5 * kl_div

    def sampler(self, mu, logvar):  
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = epsilon * std + mu
        else:
            z = mu
        return z   

    def fc(self,x1):

        pre1 = self.fc1(x1)
        dummy_pre = self.fc2(x1)

        mu = self.fc_mu(x1)
        logvar = self.fc_logvar(x1)

        return pre1,dummy_pre,mu,logvar
    
    def sampler(self, mu, logvar):   
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = epsilon * std + mu
        else:
            z = mu
        return z

    def restruct(self,x1):
        
        x12 = x1.reshape(-1, 64, 1, 1)
        x12 = F.relu(self.bn1_de(self.dconv1(x12)))
        x12 = F.relu(self.dconv2(x12))
        x12 = F.relu(self.bn2_de(self.dconv3(x12)))
        x12 = F.relu(self.dconv4(x12))
        HSI_x,lidar_x = torch.split(x12, [32, 32], dim=1)

        HSI = self.dconvhsi(HSI_x)
        Lidar = self.dconvlidar(lidar_x)

        return HSI,Lidar
        
    def forward(self, input):

        x1 = self.encoder(input)
        x1 = self.latter(x1)
        pre1,dummypre,mu,logvar = self.fc(x1)
        latent_z = self.sampler(mu, logvar)
        dist = self.distance(latent_z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes)
        x12,Lidar = self.restruct(x1)

        return pre1, x12, dummypre, latent_z, dist, kl_div ,Lidar

    def GetlossMGPL(self, pre1, x12, dummypre, latent_z, dist, kl_div,input,y):

        x1,x2 = torch.split(input, [input.shape[1]-2, 2], dim=1)

        dist_reshape = dist.reshape(len(x1), self.n_classes, self.n_sub_prototypes)  
        
        dist_class_min, _ = torch.min(dist_reshape, axis=2)  
        _, preds = torch.min(dist_class_min, axis=1)  

        y_one_hot = F.one_hot(torch.argmax(y,dim=1), num_classes=self.n_classes) 
        y_mask = y_one_hot.repeat_interleave(self.n_sub_prototypes,dim=1).bool()  
        dist_y = dist[y_mask].reshape(len(dist), self.n_sub_prototypes)   
        kl_div_y = kl_div[y_mask].reshape(len(kl_div), self.n_sub_prototypes)        
        
        q_w_z_y = F.softmax(-dist_y / self.temp_intra, dim=1)  

        criterion1 = nn.L1Loss()
        rec_loss = criterion1(x12, x1)

        kld_loss = torch.mean(torch.sum(q_w_z_y * kl_div_y, dim=1))  
        
        ent_loss = torch.mean(torch.sum(q_w_z_y * torch.log(q_w_z_y * self.n_sub_prototypes + 1e-7), dim=1))
        
        LSE_all_dist = torch.logsumexp(-dist / self.temp_inter, 1)
        LSE_target_dist = torch.logsumexp(-dist_y / self.temp_inter, 1)
        dis_loss = torch.mean(LSE_all_dist - LSE_target_dist)
        
        loss = {'dis': dis_loss, 'rec': rec_loss, 'kld': kld_loss, 'ent': ent_loss,}

        return loss

    

    
def imgDraw(label, imgName, path='./pictures', show=True):
    
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8') 
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  
    for i in range(1, numClass + 1):  
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  
    return Y_RGB


def displayClassTable(n_list, matTitle=""):

    from pandas import DataFrame
    lenth = len(n_list)  
    column = range(1, lenth + 1)
    table = {'Class': column, 'Total': [int(i) for i in n_list]}
    table_df = DataFrame(table).to_string(index=False)
    print(table_df)
    print('All available data total ' + str(int(sum(n_list))))
    print("+---------------------------------------------------+")


def listClassification(Y, matTitle=''):
    numClass = np.max(Y)  
    listClass = [] 
    for i in range(numClass):
        listClass.append(len(np.where(Y == (i + 1))[0]))
    displayClassTable(listClass, matTitle)
    return listClass
