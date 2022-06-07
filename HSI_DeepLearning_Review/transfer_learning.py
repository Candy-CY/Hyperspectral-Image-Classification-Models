import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import gc
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys


def set_params(args):
    args.batch_size1 = 100; args.epochs1 = 50
    args.batch_size2 = 100; args.epochs2 = 500
    return args

def get_model_compiled(args, inputshape, num_class):
    model = Sequential()
    if args.arch == "CNN1D":
        model.add(Conv1D(20, (24), activation='relu', input_shape=inputshape))
        model.add(MaxPooling1D(pool_size=5))
        model.add(Flatten())
        model.add(Dense(100))
    elif "CNN2D" in args.arch:
        model.add(Conv2D(50, kernel_size=(5, 5), input_shape=inputshape))
        model.add(Activation('relu'))
        model.add(Conv2D(100, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100))
    elif args.arch == "CNN3D":
        model.add(Conv3D(32, kernel_size=(5, 5, 24), input_shape=inputshape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(64, (5, 5, 16)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))
        model.add(Flatten())
        model.add(Dense(300))
    if args.arch != "CNN2D": model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss=categorical_crossentropy, optimizer=Adam(args.lr1), metrics=['accuracy']) 
    return model


def get_pretrained_model_compiled(args, inputshape, num_class):
    if args.use_transfer_learning: clf_pretrain = load_model("/tmp/best_base_model.h5")
    else: clf_pretrain = get_model_compiled(args, inputshape, num_class)
    last_layer = clf_pretrain.layers[-2].output
    x = Dense(100, activation="relu", name='salida_den1')(last_layer)
    x = Dropout(0.4)(x)
    x = Dense(100, activation="relu", name='salida_den2')(x)
    x = Dropout(0.4)(x)
    x = Dense(num_class, activation="softmax", name='salida_den3')(x)
    model = Model(inputs=clf_pretrain.inputs, outputs=x)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=8e-4), metrics=['accuracy'])
    return model



def main():
    parser = argparse.ArgumentParser(description='Transfer Learning HSI')
    parser.add_argument('--dataset1', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--dataset2', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--arch', type=str, required=True, \
            choices=["CNN1D", "CNN2D", "CNN2D40bands", "CNN3D"], \
            help='architecture (options: CNN1D, CNN2D, CNN2D40bands, CNN3D)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    
    parser.add_argument('--tr_samples', default=2, type=int, help='samples per class train set')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')

    parser.add_argument('--use_transfer_learning', action='store_true', help='Use transfer learning')
    parser.add_argument('--search_base_model', action='store_true', help='Search base model')

    parser.add_argument('--lr1', default=1e-3, type=float, help='Learning rate base model')
    parser.add_argument('--lr2', default=8e-4, type=float, help='Learning rate fine model')

    parser.add_argument('--verbosetrain1', action='store_true', help='Verbose train')
    parser.add_argument('--verbosetrain2', action='store_true', help='Verbose train')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size1', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs1', default=50, type=int, help='Number of full training cycle on the training set')
    parser.add_argument('--batch_size2', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs2', default=500, type=int, help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters: args = set_params(args)
    
    args.components  = 1 if "CNN2D" == args.arch else 40
    args.spatialsize = 1 if args.arch == "CNN1D" else 19
    if args.dataset2 in ["IP", "SV", "DIP", "DIPr"]: num_class = 16
    stats = np.ones((args.repeat, num_class+3)) * -1000.0 # OA, AA, K, Aclass
    for pos in range(args.repeat):
        rstate = args.random_state+pos if args.random_state != None else None
        if args.search_base_model:
            pixels, labels, num_class = \
                            mydata.loadData(args.dataset1, num_components=args.components, preprocessing=args.preprocess)

            if args.arch != "CNN1D":
                pixels, labels = mydata.createImageCubes(pixels, labels, windowSize=args.spatialsize, removeZeroLabels = True)
                if args.arch == "CNN3D":
                    inputshape = (pixels.shape[1], pixels.shape[2], pixels.shape[3], 1)
                    pixels = pixels.reshape(pixels.shape[0], pixels.shape[1], pixels.shape[2], pixels.shape[3], 1)
                else: inputshape = (pixels.shape[1], pixels.shape[2], pixels.shape[3])
            else:
                pixels = pixels.reshape(-1, pixels.shape[-1])
                labels = labels.reshape(-1)
                pixels = pixels[labels!=0]
                labels = labels[labels!=0] - 1
                inputshape = (pixels.shape[-1], 1)
                pixels = pixels.reshape(pixels.shape[0], pixels.shape[1], 1)

            pixels, labels = mydata.random_unison(pixels, labels, rstate=rstate)
            clf = get_model_compiled(args, inputshape, num_class)
            clf.fit(pixels, keras_to_categorical(labels),
                            batch_size=args.batch_size1,
                            epochs=args.epochs1,
                            verbose=args.verbosetrain1,
                            callbacks = [ModelCheckpoint("/tmp/best_base_model.h5", monitor='loss', verbose=0, save_best_only=True)])
            del pixels, labels
            del clf; K.clear_session(); gc.collect()
            exit()

        pixels, labels, num_class = \
                        mydata.loadData(args.dataset2, num_components=args.components, preprocessing=args.preprocess)

        if args.arch != "CNN1D": 
            pixels, labels = mydata.createImageCubes(pixels, labels, windowSize=args.spatialsize, removeZeroLabels = False)
            if args.arch == "CNN3D":
                inputshape = (pixels.shape[1], pixels.shape[2], pixels.shape[3], 1)
                pixels = pixels.reshape(pixels.shape[0], pixels.shape[1], pixels.shape[2], pixels.shape[3], 1)
            else: inputshape = (pixels.shape[1], pixels.shape[2], pixels.shape[3])
        else:
            pixels = pixels.reshape(-1, pixels.shape[-1])
            labels = labels.reshape(-1)
            pixels = pixels[labels!=0]
            labels = labels[labels!=0] - 1
            inputshape = (pixels.shape[-1], 1)
            pixels = pixels.reshape(pixels.shape[0], pixels.shape[1], 1)

        if args.dataset2 in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
            x_train, x_test, y_train, y_test = \
                mydata.load_split_data_fix(args.dataset, pixels)
        else:
            pixels = pixels[labels!=0]
            labels = labels[labels!=0] - 1
            x_train, x_test, y_train, y_test = \
                mydata.split_data(pixels, labels, [args.tr_samples]*num_class, splitdset="custom2", rand_state=rstate)
        if args.use_val:
            x_val, x_test, y_val, y_test = \
                mydata.split_data(x_test, y_test, args.val_percent, rand_state=rstate)
            if args.arch == "CNN1D": x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            elif args.arch == "CNN3D":
                x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1)
        if args.arch == "CNN1D":
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        elif args.arch == "CNN3D": 
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
            x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)
            inputshape = (pixels.shape[1], pixels.shape[2], pixels.shape[3], 1)

        valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (x_test, keras_to_categorical(y_test, num_class))
        clf = get_pretrained_model_compiled(args, inputshape, num_class)

        clf.fit(x_train, keras_to_categorical(y_train),
                batch_size=args.batch_size2,
                epochs=args.epochs2,
                verbose=args.verbosetrain2,
                validation_data=valdata,
                callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
        del clf; K.clear_session(); gc.collect()
        clf = load_model("/tmp/best_model.h5")
        #print("PARAMETERS", clf.count_params())
        stats[pos,:] = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print(stats[-1])

if __name__ == '__main__':
    main()



























