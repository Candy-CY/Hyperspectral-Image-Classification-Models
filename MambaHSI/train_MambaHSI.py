import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import time
import torch
import random
import argparse
import numpy as np
from torchvision import models,transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss,resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching

# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
from model.MambaHSI import MambaHSI

from calflops import calculate_flops

torch.autograd.set_detect_anomaly(True)

time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int,default=0)
    parser.add_argument('--data_set_path',type=str,default='./data')
    parser.add_argument('--work_dir',type=str,default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost',type=bool,default=True)

    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
seed_list = [0,1,2,3,4,5,6,7,8,9]  #
# seed_list = [0]  #

num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index

max_epoch = args.max_epoch
learning_rate = args.lr

net_name = 'MambaHSI'

paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list}


                      # 0        1         2         3        4
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston']
data_set_name = data_set_name_list[dataset_index]

if data_set_name in ['HanChuan','Houston']:
    split_image = True
else:
    split_image = False

transform = transforms.Compose([
    # transforms.Resize((2048, 1024)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
])


if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(args.train_samples),str(args.val_samples)) + '_lr{}'.format(str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder,'train_tr{}_val{}.log'.format(num_list[0],num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name),logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    height, width, channels = data.shape

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    img = ImageStretching(data)

    class_count = max(np.unique(gt))

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio]

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx,curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                       num_list,
                                                                                                       gt_reshape,
                                                                                                       class_count,
                                                                                                       flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data, height, width, gt_reshape, index)

        # build Model

        net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=128)
        logger.info(paras_dict)
        logger.info(net)

        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(device)

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        logger.info(optimizer)
        best_loss = 99999
        if record_computecost:
            net.eval()
            flops, macs1, para = calculate_flops(model=net,
                                                 input_shape=(1, x.shape[1], x.shape[2], x.shape[3]), )
            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0


        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()
            loss_dict = {}

            net.train()

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2+5, :]
                y_part1 = y_train[:,:x.shape[2] // 2+5,:]
                x_part2 = x[:, :, x.shape[2] // 2 - 5: , :]
                y_part2 = y_train[:,x.shape[2] // 2 - 5:,:]
                y_pred_part1 = net(x_part1)

                ls1 = head_loss(loss_func,y_pred_part1, y_part1.long())
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func,y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            else:
                try:
                    y_pred = net(x)
                    ls = head_loss(loss_func,y_pred, y_train.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    logger.info('Iter:{}|loss:{}'.format(epoch, ls.detach().cpu().numpy()))
                except:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image=True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    y_pred_part1 = net(x_part1)
                    ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    y_pred_part2 = net(x_part2)
                    ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    logger.info(
                        'Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            torch.cuda.empty_cache()
            # evaluate stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                # output_val = net(x)
                output_val = net(x)
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val,
                                    size=y_val.shape[1:],
                                    mode='bilinear',
                                    align_corners=True)
                predict = torch.argmax(seg_logits,dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255,axis=0),predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA,mAcc,Kappa,mIOU,IOU,Acc))
                # save weight
                if OA>=best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    # torch.save(net,save_weight_path)
                    torch.save(net.state_dict(), save_weight_path)
                    # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                    # torch.save(net.state_dict(), save_epoch_weight_path)
                if (epoch+1)%50==0:
                    save_single_predict_path = os.path.join(save_vis_folder,'predict_{}.png'.format(str(epoch+1)))
                    save_single_gt_path = os.path.join(save_vis_folder,'gt.png')
                    vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)

                # net.train()
            torch.cuda.empty_cache()


        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        # best_net = copy.deepcopy(net)
        best_net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=128)

        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            test_evaluator.reset()
            output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test,
                                size=y_test.shape[1:],
                                mode='bilinear',
                                align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                    Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', list(OA_ALL))
    logger.info('List of AA:', list(AA_ALL))
    logger.info('List of KPP:', list(KPP_ALL))
    logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
          np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

    # Output infors
    mean_result_path = os.path.join(save_folder,'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(
        np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(
        np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()

    del net
