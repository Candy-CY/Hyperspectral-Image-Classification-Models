import scipy.io as sio
from matplotlib import cm
from sklearn import metrics
import time
from sklearn import preprocessing
from utils import *
from SubGraphSampler import SubGraphSampler
from torch_geometric.data import DataLoader
from Networks import *
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)
torch.cuda.manual_seed_all(2022)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" )
torch.autograd.set_detect_anomaly(True)

## get classification map
CLASSIFICATIONMAP=0

## show similarity between prototypes
INTER_SIMILARITY=0

classifier=["softmax","metric"][  1  ]

##### Hyperparameters
max_epoch = 200
patch_radius = 4
num_hops = 1
n_avgsize = 64
knn_neighbors = 25
network_depth = 5
# for (FLAG, curr_train_ratio,Scale) in [(1,0.1,100),(1,0.1,200),(1,0.1,300),(1,0.1,400),(1,0.1,500),
# (2,0.01,100),(2,0.01,200),(2,0.01,300),(2,0.01,400),(2,0.01,500),
# (3,0.01,100),(3,0.01,200),(3,0.01,300),(3,0.01,400),(3,0.01,500)]:
for (FLAG, curr_train_ratio) in [(1,0.05),(2,0.005),(3,0.005),(5,0.05)]:
    OA_ALL = []; AA_ALL = []; KPP_ALL = []; AVG_ALL = []; Train_Time_ALL=[]; Test_Time_ALL=[]
    samples_type = 'ratio' if curr_train_ratio < 1 else 'same_num'
    # Seed_List=[0]
    # Seed_List=[0,1,2,3,4]
    Seed_List = [0,1,2,3,4,5,6,7,8,9]
    
    if FLAG == 1:
        data_mat = sio.loadmat('hsi_data\\Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('hsi_data\\Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        
        val_ratio = 0.01
        class_count = 16
        learning_rate = 5e-4
        dataset_name = "indian_"
    
    if FLAG == 2:
        data_mat = sio.loadmat('hsi_data\\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('hsi_data\\Pavia_University_gt.mat')
        gt = gt_mat['pavia_university_gt']

        val_ratio = 0.005
        class_count = 9
        learning_rate = 5e-4
        dataset_name = "paviaU_"
        pass

    if FLAG == 3:
        data_mat = sio.loadmat('hsi_data\\Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('hsi_data\\Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        
        val_ratio = 0.005
        class_count = 16
        learning_rate = 5e-4
        dataset_name = "salinas_"
        pass
    
    if FLAG == 4:
        data_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC_gt.mat')
        gt = gt_mat['KSC_gt']

        val_ratio = 0.01
        class_count = 13
        learning_rate = 1e-3
        dataset_name = "KSC_"
        pass

    if FLAG == 5:
        data_mat = sio.loadmat('hsi_data\\Loukia.mat')
        data = data_mat['Loukia']
        gt_mat = sio.loadmat('hsi_data\\Loukia_GT.mat')
        gt = gt_mat['Loukia_GT']

        val_ratio = 0.01
        class_count = 14
        learning_rate = 5e-4
        dataset_name = "Loukia_"
        pass

    ## for evaluation with same number of samples per class
    train_samples_per_class = curr_train_ratio
    if samples_type=='same_num': val_ratio = 1 # curr_train_ratio
    
    train_ratio = curr_train_ratio
    cmap = cm.get_cmap('jet', 10)
    plt.set_cmap(cmap)
    m, n, d = data.shape

    # standardization
    orig_data=data
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])


    # show numbers of samples
    gt_reshape = np.reshape(gt, [-1])
    samplesCount_perclass = []
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount_perclass.append(len(idx))
    print(samplesCount_perclass)
    

    for curr_seed in Seed_List:
        train_samples_gt, test_samples_gt, val_samples_gt = get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio,
                                                                           val_ratio, samples_type)
        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        # show numbers of samples (training, validation, testing)
        train_val_test_gt = [train_samples_gt, val_samples_gt, test_samples_gt]
        for type in range(3):
            gt_reshape = np.reshape(train_val_test_gt[type], [-1])
            print("===============================")
            samplesCount_list = []
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                samplesCount_list.append(samplesCount)
            print(samplesCount_list)
    
        ############Generating the sample index of training, validation, and testing data sets##############
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])
    
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])
    
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])
        
        
        
        tic = time.perf_counter()

        print('dataset loading ... ')
        # Composite Neighbors Generator
        GD = GetDataset(data,extend=patch_radius)
        trainDatasetList=GD.getSamples(train_samples_gt_onehot)
        valDatasetList=GD.getSamples(val_samples_gt_onehot)
        
        SS=SubGraphSampler(data)
        SS.ConstructGraph(knn_neighbors=knn_neighbors,n_avgsize=n_avgsize,show_segmap=True)
        trainGraphsList=SS.get_graph_samples(gt=train_samples_gt_onehot,num_hops=num_hops)
        valGraphsList=SS.get_graph_samples(gt=val_samples_gt_onehot,num_hops=num_hops)
        
        # transfer data to GPU
        trainDatasetList_gpu=[];trainGraphsList_gpu=[]
        valDatasetList_gpu=[];valGraphsList_gpu=[]

        print('transfer dataset to GPU ... ')
        
        for s in trainDatasetList:
            sample=s[0].to(device)
            label=s[1].to(device)
            
            trainDatasetList_gpu.append([sample,label])
        
        for s in trainGraphsList:
            trainGraphsList_gpu.append(s.to(device))
            
        for s in valDatasetList:
            sample = s[0].to(device)
            label = s[1].to(device)
            valDatasetList_gpu.append([sample,label])
            
        for s in valGraphsList:
            valGraphsList_gpu.append(s.to(device))
        

        
        trainData=PatchDataset(trainDatasetList_gpu,trainGraphsList_gpu,is_Already_to_gpu=True,device=device)
        valData=PatchDataset(valDatasetList_gpu,valGraphsList_gpu,is_Already_to_gpu=True,device=device)
        
        trainDataLoder =DataLoader(dataset=trainData, batch_size=64, shuffle=True ,num_workers=0,drop_last=False)
        valDataLoder =DataLoader(dataset=valData, batch_size=64, shuffle=False ,num_workers=0)
        
        
        ## network
        net = CNCMN(patch_radius*2+1, patch_radius*2+1, bands, class_count,learning_rate=learning_rate,classifier=classifier,netDepth=network_depth)
        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)

        def evaluate_performance(network_output,samples_gt, require_AA_KPP=False,printFlag=True):
            if False==require_AA_KPP:
                with torch.no_grad():
                    available_label_count = samples_gt.shape[0]
                    correct_prediction = \
                    torch.where(torch.argmax(network_output, -1) == torch.argmax(samples_gt, -1))[0].shape[0]
                    OA = correct_prediction / available_label_count
                    return OA
            if require_AA_KPP:
                with torch.no_grad():
                    #OA
                    available_label_count = samples_gt.shape[0]
                    correct_prediction = \
                    torch.where(torch.argmax(network_output, -1) == torch.argmax(samples_gt, -1))[0].shape[0]
                    OA = correct_prediction / available_label_count

                    # AA
                    AA_list = []
                    samples_perclass=torch.sum(samples_gt,0).cpu().numpy()
                    for i in range(class_count):
                        correct_prediction = torch.where((torch.argmax(network_output, -1) == torch.argmax(samples_gt, -1)) & (samples_gt[:,i]==1))[0].shape[0]
                        AA_ = correct_prediction / samples_perclass[i]
                        AA_list.append(AA_)
                    AA=np.mean(AA_list)

                    # kappa
                    kappa = metrics.cohen_kappa_score(torch.argmax(network_output, -1).detach().cpu().numpy(),
                                                      torch.argmax(samples_gt, -1).detach().cpu().numpy())
                    
                    
                    if printFlag:
                        print("test OA=", OA, "AA=", AA, 'kpp=', kappa)
                        print('acc per class:')
                        print(AA_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(AA)
                    KPP_ALL.append(kappa)
                    AVG_ALL.append(AA_list)

                    # 保存数据信息
                    f = open('results\\' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(AA) \
                                  + '\nkpp=' + str(kappa) \
                                  + '\nacc per class:' + str(AA_list) + "\n"
                                  # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                  # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA
        
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        
        '''Training'''
        best_loss_1=99999
        tic1 = time.perf_counter()
        for epoch in range(max_epoch):
            net.train()
            trainloss = 0
            network_output = []
            real_label = []
            for inputs, labels,subgraphs in trainDataLoder:
                optimizer.zero_grad()
                [output,loss] ,_= net(inputs,labels,subgraphs)
                trainloss = trainloss + loss
                loss.backward(retain_graph=False)
                optimizer.step()  # Does the update
                network_output.append(output)
                real_label.append(labels)
            trainOA = evaluate_performance(torch.cat(network_output, 0), torch.cat(real_label, 0))
            with torch.no_grad():
                net.eval()
                valloss = 0
                valloss_all=0
                network_output = []
                real_label = []
                for inputs, labels,subgraphs in valDataLoder:
                    [output, loss],_ = net(inputs,labels,subgraphs)
                    if classifier=="softmax": loss = F.cross_entropy(output, torch.argmax(labels, dim=-1))
                    valloss =valloss+ loss
                    network_output.append(output)
                    real_label.append(labels)
                valOA = evaluate_performance(torch.cat(network_output, 0), torch.cat(real_label, 0))
                print('epoch:', epoch, 'train loss=', trainloss.cpu().numpy(),
                      'train OA=', trainOA,
                      'val loss=', valloss.cpu().numpy(),
                      'val OA=', valOA, )
                if (valloss < best_loss_1) or (valloss == best_loss_1):
                    best_loss_1 = valloss
                    print("save model...")
                    torch.save(net.state_dict(), "model\\best_model.pt")
                    pass
                
                # show current similarities between prototypes
                if classifier=="metric" and INTER_SIMILARITY==1 and epoch%20==0:
                    capsules = net.getCapsules().detach().cpu().numpy()
                    M = np.maximum( np.matmul(capsules,np.transpose(capsules)), 0)
                    fig=plt.figure()
                    plt.imshow(M)
                    plt.xlim((-0.5, class_count-0.5))
                    plt.ylim((-0.5, class_count-0.5))
                    ax = plt.gca()
                    my_x_ticks = np.arange(0, class_count,1)
                    my_y_ticks = np.arange(0, class_count,1)
                    plt.xticks(my_x_ticks,my_x_ticks+1)
                    plt.yticks(my_y_ticks,my_y_ticks+1)
                    ax.xaxis.tick_top()
                    ax.invert_yaxis()
                    plt.show()
                    
        
        trainTime = time.perf_counter()-tic1
        Train_Time_ALL.append(trainTime)
        
        # # Testing
        testDatasetList=GD.getSamples(test_samples_gt_onehot,rotation=0)
        testGraphsList = SS.get_graph_samples(gt=test_samples_gt_onehot, num_hops=num_hops)

        testData = PatchDataset(testDatasetList,testGraphsList, is_Already_to_gpu=False, device=device)
        testDataLoder =DataLoader(dataset=testData, batch_size=64, shuffle=False ,num_workers=0)
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            
            tic2 = time.perf_counter()
            network_output=[]
            embeddings_all=[]
            real_label=[]
            for inputs, labels,subgraphs in testDataLoder:
                [output,_],embeddings = net(inputs,None,subgraphs)
                network_output.append(output)
                embeddings_all.append(embeddings)
                real_label.append(labels)
            testOA = evaluate_performance(torch.cat(network_output,0),torch.cat(real_label,0).to(device),require_AA_KPP=True)

            testTime = time.perf_counter() - tic2
            Test_Time_ALL.append(testTime)
            

        '''get classification map'''
        if CLASSIFICATIONMAP==1:
            all_samples_gt_onehot=np.ones_like(test_samples_gt_onehot)
            allDatasetList = GD.getSamples(all_samples_gt_onehot, rotation=0)
            allGraphsList = SS.get_graph_samples(gt=all_samples_gt_onehot, num_hops=num_hops)
            allData = PatchDataset(allDatasetList, allGraphsList,is_Already_to_gpu=False, device=device)
            allDataLoder = DataLoader(dataset=allData, batch_size=64, shuffle=False, num_workers=0)
            torch.cuda.empty_cache()
            with torch.no_grad():
                torch.cuda.empty_cache()
                net.load_state_dict(torch.load("model\\best_model.pt"))
                net.eval()
                network_output = []
                for inputs, _,subgraphs in allDataLoder:
                    [output,_],embeddings = net(inputs,None,subgraphs)
                    predict=torch.argmax(output, -1).cpu().numpy()+1
                    network_output=np.concatenate([network_output,predict])
                network_output=np.reshape(network_output,[height, width])
                Draw_Classification_Map(network_output,"results\\"+dataset_name+str(testOA))
        
        del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)
    #
    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    # save info
    f = open('results\\' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
    +"\ntrain_ratio={}".format(curr_train_ratio) \
    +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
    +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()
        

    
    
    
    
    
    
    