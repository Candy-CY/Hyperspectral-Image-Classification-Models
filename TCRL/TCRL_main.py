import argparse
from utils import load_config, get_log_name, set_seed, \
    plot_results, record, aa_and_each_accuracy, Draw_Classification_Map

import algorithms
import numpy as np
import nni
import collections
import datetime
from sklearn import metrics
from HSI_dataset import data_processing
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./configs/tcrl.py',
                    help='The path of config file.')
args = parser.parse_args()


def main():
    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=True)
    config.update(tuner_params)

    # load data
    all_iter, BAND, CLASSES_NUM, data_shape = data_processing(Dataset=config['dataset'], batch_size=config['batch_size'],
                                                  PATCH_LENGTH=config['patch_length'],
                                                  flag='testAll')
    seeds = [0]

    overall_acc, aver_acc, KAPPA = [], [], []
    ELEMENT_ACC = np.zeros((len(seeds), CLASSES_NUM))
    count = 0
    for seed in seeds:
        day = datetime.datetime.now()
        day_str1 = day.strftime('%m_%d_%H_%M')
        set_seed(seed)
        # load dataset

        train_loader, valida_loader, test_loader = data_processing(Dataset=config['dataset'], batch_size=config['batch_size'],
                                                                   PATCH_LENGTH=config['patch_length'],
                                                                   Train_size=config['train_size'],
                                                                   flag='train', noise_type=config['noise_type'],
                                                                   noise_ratio=config['percent'])
        num_test_images = len(valida_loader.dataset)
        model = algorithms.__dict__[config['algorithm']](config, input_channel=BAND, num_classes=CLASSES_NUM)
        epoch = 0
        # evaluate models with random weights
        test_acc, _, _ = model.evaluate(valida_loader)
        print('Run %d, Epoch [%d/%d] Test Accuracy on the %s test images: %.4f' % (
            count+1, epoch + 1, config['epochs'], num_test_images, test_acc))

        # train and test
        acc_all_list,train_acc_list = [],[]
        for epoch in range(1, config['epochs']):
            # train
            Tr_acc = model.train(train_loader, epoch)
            # validation
            test_acc, pre1, gt_test = model.evaluate(valida_loader)
            nni.report_intermediate_result(test_acc)
            collections.Counter(pre1)
            confusion_matrix = metrics.confusion_matrix(pre1, gt_test)
            each_acc1, aa1 = aa_and_each_accuracy(confusion_matrix)
            Kappa1 = metrics.cohen_kappa_score(pre1, gt_test)

            print('Run %d, Epoch [%d/%d] Test Accuracy on the %s test images: %.2f  AA %.2f  Kappa %.2f ' % (
                count+1, epoch + 1, config['epochs'], num_test_images, test_acc, aa1*100, Kappa1*100))

            acc_all_list.extend([test_acc])
            train_acc_list.extend([Tr_acc])
        # test, just depend on the model of the last epoch
        acc1, pre1, gt_test = model.evaluate(test_loader)
        collections.Counter(pre1)
        OA = metrics.accuracy_score(pre1, gt_test)
        confusion_matrix = metrics.confusion_matrix(pre1, gt_test)
        Each_acc, AA = aa_and_each_accuracy(confusion_matrix)
        Kappa = metrics.cohen_kappa_score(pre1, gt_test)
        print('OA, AA, Kappa:', OA, AA, Kappa)
        print(Each_acc)
        KAPPA.append(Kappa)
        overall_acc.append(OA)
        aver_acc.append(AA)
        ELEMENT_ACC[count, :] = Each_acc
        count += 1

        nni.report_final_result(OA)
        print('  ********************************************  ')
        # Plot the training/test acc vs epoch
        # jsonfile = get_log_name(day_str1, config)
        # plot_results(epochs=config['epochs'], test_acc=acc_all_list, plotfile=jsonfile.replace('.json', '.png'))
        # plot_results(epochs=config['epochs'], test_acc=train_acc_list, plotfile=jsonfile.replace('.json', '.png'))

    # Draw the classification map
    _, pre1, _ = model.evaluate(all_iter)
    pre1 = np.array(pre1).reshape((data_shape[0], data_shape[1]))
    Draw_Classification_Map(pre1 + 1,
                            'log/' + config['dataset'] + day_str1 + '_Train_' + str(
                                len(train_loader.dataset)) + '_' +
                            config['algorithm'] + '_' + config['noise_type'] + '_NumOfNoisyLabel_' + \
                            str(config['percent']) + '_batchSize_' + str(config['batch_size']))

    # Save the average accuracy to .txt
    if config['save_result']:
        record.record_output(overall_acc, aver_acc, KAPPA, ELEMENT_ACC,
                             'log/' + config['dataset'] + day_str + '_Train_' + str(len(train_loader.dataset)) + '_' +
                             config['algorithm'] + '_' + config['noise_type'] + '_NumOfNoisyLabel_' + \
                             str(config['percent']) + '_batchSize_' + str(config['batch_size']) + '.txt')


if __name__ == '__main__':
    main()
