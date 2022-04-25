import os
import time
import argparse
from PIL import Image
import torch
from torch.autograd import Variable
from HyperTools import *
from Models import *

DataName = {1:'Pavia',2:'Salinas'}

def main(args):
    if args.dataID==1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/Pavia/'
    elif args.dataID==2:       
        num_classes = 16  
        num_features = 204  
        save_pre_dir = './Data/Salinas/'

    X = np.load(save_pre_dir+'X.npy')
    _,h,w = X.shape
    Y = np.load(save_pre_dir+'Y.npy')
    
    X_train = np.reshape(X,(1,num_features,h,w))
    train_array = np.load(save_pre_dir+'train_array.npy')
    Y_train = np.ones(Y.shape)*255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train,(1,h,w)) 

    # define the targeted label in the attack
    Y_tar = np.zeros(Y.shape)
    Y_tar = np.reshape(Y_tar,(1,h,w))

    save_path_prefix = args.save_path_prefix+'Exp_'+DataName[args.dataID]+'/'
    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)
    
    num_epochs = 1000    
    if args.model=='SACNet':    
        Model = SACNet(num_features=num_features,num_classes=num_classes)
    elif args.model=='DilatedFCN ':
        Model = DilatedFCN (num_features=num_features,num_classes=num_classes)
    elif args.model=='SpeFCN':
        Model = SpeFCN(num_features=num_features,num_classes=num_classes)
        num_epochs = 3000
    elif args.model=='SpaFCN':
        Model = SpaFCN(num_features=num_features,num_classes=num_classes)
    elif args.model=='SSFCN':
        Model = SSFCN(num_features=num_features,num_classes=num_classes)
    Model = torch.nn.DataParallel(Model).cuda()
    Model.train()
    optimizer = torch.optim.Adam(Model.parameters(),lr=args.lr,weight_decay=args.decay)

    images = torch.from_numpy(X_train).float().cuda()
    label = torch.from_numpy(Y_train).long().cuda()
    criterion = CrossEntropy2d().cuda()      

    # train the classification model
    for epoch in range(num_epochs):  
        adjust_learning_rate(optimizer,args.lr,epoch,num_epochs)
        tem_time = time.time()      
        optimizer.zero_grad()
        output = Model(images)  
                
        seg_loss = criterion(output,label)
        seg_loss.backward()

        optimizer.step()
       
        batch_time = time.time()-tem_time
        if (epoch+1) % 1 == 0:            
            print('epoch %d/%d:  time: %.2f cls_loss = %.3f'%(epoch+1, num_epochs,batch_time,seg_loss.item()))
    
    Model.eval()
    output = Model(images)  
    _, predict_labels = torch.max(output, 1)  
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)

    # adversarial attack
    epsilon = [0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10]
    for i in range(len(epsilon)):
        print('Generate adversarial example with epsilon = %.2f'%(epsilon[i]))
        processed_image = Variable(images)
        processed_image = processed_image.requires_grad_()
        label = torch.from_numpy(Y_tar).long().cuda()
                                                                    
        output = Model(processed_image)
        seg_loss = criterion(output,label)
        seg_loss.backward()
        
        adv_noise = epsilon[i] * processed_image.grad.data / torch.norm(processed_image.grad.data,float("inf"))

        processed_image.data = processed_image.data - adv_noise
       
        X_adv = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
        noise_image = X_adv - images.cpu().data.numpy()[0]        
        noise_image[noise_image > 1] = 1
        noise_image[noise_image < 0] = 0  

        if args.dataID == 1:
            im = Image.fromarray(np.moveaxis((noise_image[[102,56,31],:,:]*25500).astype('uint8'),0,-1))
            im.save(save_path_prefix+'perturbation'+str(epsilon[i])+'.png','png')
            im = Image.fromarray(np.moveaxis((X_adv[[102,56,31],:,:]*255).astype('uint8'),0,-1))
            im.save(save_path_prefix+'advimage'+str(epsilon[i])+'.png','png')
        elif args.dataID == 2:
            im = Image.fromarray(np.moveaxis((noise_image[[102,56,31],:,:]*25500).astype('uint8'),0,-1))
            im.save(save_path_prefix+'perturbation'+str(epsilon[i])+'.png','png')
            im = Image.fromarray(np.moveaxis((X_adv[[102,56,31],:,:]*255).astype('uint8'),0,-1))
            im.save(save_path_prefix+'advimage'+str(epsilon[i])+'.png','png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='SACNet')
    
    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)

    main(parser.parse_args())
