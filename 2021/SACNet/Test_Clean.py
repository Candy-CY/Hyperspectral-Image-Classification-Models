import os
import time
import argparse
import torch
from HyperTools import *
from Models import *

DataName = {1:'PaviaU',2:'Salinas'}

def main(args):
    if args.dataID==1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/PaviaU/'
    elif args.dataID==2:
        num_classes = 16  
        num_features = 204  
        save_pre_dir = './Data/Salinas/'

    X = np.load(save_pre_dir+'X.npy')
    _,h,w = X.shape
    Y = np.load(save_pre_dir+'Y.npy')
    
    X_train = np.reshape(X,(1,num_features,h,w))
    train_array = np.load(save_pre_dir+'train_array.npy')
    test_array = np.load(save_pre_dir+'test_array.npy')
    Y_train = np.ones(Y.shape)*255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train,(1,h,w)) 
   

    save_path_prefix = args.save_path_prefix+'Exp_'+DataName[args.dataID]+'/'
    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)
    
    num_epochs = 1000   
    if args.model=='SACNet':     
        Model = SACNet(num_features=num_features,num_classes=num_classes)    
    elif args.model=='DilatedFCN':
        Model = DilatedFCN(num_features=num_features,num_classes=num_classes)
    elif args.model=='SpeFCN':
        Model = SpeFCN(num_features=num_features,num_classes=num_classes)
        num_epochs = 3000
    elif args.model=='SpaFCN':
        Model = SpaFCN(num_features=num_features,num_classes=num_classes)
    elif args.model=='SSFCN':
        Model = SSFCN(num_features=num_features,num_classes=num_classes)
   
    Model = Model.cuda()
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

    # results on the clean test set
    OA,kappa,ProducerA = CalAccuracy(predict_labels[test_array],Y[test_array])

    img = DrawResult(np.reshape(predict_labels+1,-1),args.dataID)
    plt.imsave(save_path_prefix+args.model+'_clean_OA'+repr(int(OA*10000))+'_kappa'+repr(int(kappa*10000))+'.png',img)

    print('OA=%.3f,Kappa=%.3f' %(OA*100,kappa*100))
    print('producerA:',ProducerA)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='SACNet')
    
    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)

    main(parser.parse_args())
