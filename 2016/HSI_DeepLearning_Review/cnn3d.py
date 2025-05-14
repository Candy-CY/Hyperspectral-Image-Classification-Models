import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import gc
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.layers import Activation, BatchNormalization, Conv3D, Dense, Flatten, MaxPooling3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys


def set_params(args):
    args.batch_size = 100; args.epochs = 100
    return args

def get_model_compiled(shapeinput, num_class, w_decay=0, lr=1e-3):
    clf = Sequential()
    clf.add(Conv3D(32, kernel_size=(5, 5, 24), input_shape=shapeinput))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Conv3D(64, (5, 5, 16)))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling3D(pool_size=(2, 2, 1)))
    clf.add(Flatten())
    clf.add(Dense(300, kernel_regularizer=regularizers.l2(w_decay)))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Dense(num_class, activation='softmax'))
    clf.compile(loss=categorical_crossentropy, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return clf

def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=40, type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=19, type=int, help='windows size')
    parser.add_argument('--wdecay', default=0, type=float, help='apply penalties on layer parameters')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.add_argument('--verbosetrain', action='store_true', help='Verbose train')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters: args = set_params(args)

    pixels, labels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    pixels, labels = mydata.createImageCubes(pixels, labels, windowSize=args.spatialsize, removeZeroLabels = False)
    stats = np.ones((args.repeat, num_class+3)) * -1000.0 # OA, AA, K, Aclass
    for pos in range(args.repeat):
        rstate = args.random_state+pos if args.random_state != None else None
        if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
            x_train, x_test, y_train, y_test = \
                mydata.load_split_data_fix(args.dataset, pixels)#, rand_state=args.random_state+pos)
        else:
            pixels = pixels[labels!=0]
            labels = labels[labels!=0] - 1
            x_train, x_test, y_train, y_test = \
                mydata.split_data(pixels, labels, args.tr_percent, rand_state=rstate)

        if args.use_val:
            x_val, x_test, y_val, y_test = \
                mydata.split_data(x_test, y_test, args.val_percent, rand_state=rstate)
            x_val   = x_val[..., np.newaxis]
        x_test  = x_test[..., np.newaxis]
        x_train = x_train[..., np.newaxis]

        inputshape = x_train.shape[1:]
        clf = get_model_compiled(inputshape, num_class, w_decay=args.wdecay, lr=args.lr)
        valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (x_test, keras_to_categorical(y_test, num_class))
        clf.fit(x_train, keras_to_categorical(y_train, num_class),
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=args.verbosetrain,
                        validation_data=valdata,
                        callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
        del clf; K.clear_session(); gc.collect()
        clf = load_model("/tmp/best_model.h5")
        print("PARAMETERS", clf.count_params())
        stats[pos,:] = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print(args.dataset, list(stats[-1]))

if __name__ == '__main__':
    main()



























