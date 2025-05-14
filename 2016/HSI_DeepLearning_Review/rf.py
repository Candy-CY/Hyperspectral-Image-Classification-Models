import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
import sys


def set_params(args):
    if args.dataset   in ["IP", "DIP", "DIPr"]:    args.n_est = 200; args.m_s_split = 3; args.max_feat = 10; args.depth = 10
    elif args.dataset in ["UP", "DUP", "DUPr"]:  args.n_est = 200; args.m_s_split = 2; args.max_feat = 40; args.depth = 60
    elif args.dataset == "SV":  args.n_est = 200; args.m_s_split = 2; args.max_feat = 10; args.depth = 10
    elif args.dataset == "UH":  args.n_est = 200; args.m_s_split = 2; args.max_feat = 8; args.depth = 20
    return args


def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')

    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')

    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--n_est', default=200, type=int, help='The number of trees in the forest')
    parser.add_argument('--m_s_split', default=2, type=int,
                    help='The minimum number of samples required to split an internal node')
    parser.add_argument('--max_feat', default=40, type=int, 
                    help='The number of features to consider when looking for the best split')
    parser.add_argument('--depth', default=60, type=int, help='The maximum depth of the tree')
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
        clf = RandomForestClassifier(\
                    n_estimators=args.n_est, min_samples_split=args.m_s_split, \
                    max_features=args.max_feat, max_depth=args.depth) \
                    .fit(x_train, y_train)
        stats[pos,:] = mymetrics.reports(clf.predict(x_test), y_test)[2]
    print(stats[-1])


if __name__ == '__main__':
    main()

