import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import numpy as np
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
import sys


def set_params(args):
    if args.dataset   in ["IP", "DIP", "DIPr"]:    args.C = 1e2; args.g = 0.125
    elif args.dataset in ["UP", "DUP", "DUPr"]:  args.C = 1e3; args.g = 2
    elif args.dataset == "SV":  args.C = 1e3; args.g = 0.25
    elif args.dataset == "UH":  args.C = 1e5; args.g = 0.03125
    return args


def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--preprocess', default="minmax", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--C', default=1, type=int, help='Inverse of regularization strength')
    parser.add_argument('--g', default=1, type=float, help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters: args = set_params(args)

    pixels, labels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    pixels = pixels.reshape(-1, pixels.shape[-1])

    stats = np.ones((args.repeat, num_class+3)) * -1000.0 # OA, AA, K, Aclass
    for pos in range(args.repeat):
        if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
            x_train, x_test, y_train, y_test = \
                mydata.load_split_data_fix(args.dataset, pixels)#, rand_state=args.random_state+pos)
        else:
            labels = labels.reshape(-1)
            pixels = pixels[labels!=0]
            labels = labels[labels!=0] - 1
            rstate = args.random_state+pos if args.random_state != None else None
            x_train, x_test, y_train, y_test = \
                mydata.split_data(pixels, labels, args.tr_percent, rand_state=rstate)
        clf = SVC(gamma=args.g, C=args.C, tol=1e-7).fit(x_train, y_train)
        stats[pos,:] = mymetrics.reports(clf.predict(x_test), y_test)[2]
    print(stats[-1])

if __name__ == '__main__':
    main()

