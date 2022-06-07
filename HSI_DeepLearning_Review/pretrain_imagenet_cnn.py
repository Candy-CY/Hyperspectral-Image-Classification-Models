import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import cv2
import gc
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
#from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.applications import densenet, inception_v3, mobilenet, resnet50, vgg16, vgg19, xception
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys


def set_params(args):
    args.batch_size = 100; args.epochs = 200
    return args


def get_model_pretrain(arch):
    modlrate = 1
    if   "VGG16" in arch:       base_model = vgg16.VGG16
    elif "VGG19" in arch:       base_model = vgg19.VGG19
    elif "RESNET50" in arch:    base_model = resnet50.ResNet50
    elif "DENSENET121" in arch: base_model = densenet.DenseNet121
    elif "MOBILENET" in arch:
        base_model = mobilenet.MobileNet
        modlrate = 10
    else: print("model not avaiable"); exit()
    base_model = base_model(weights='imagenet', include_top=False)
    return base_model, modlrate


def get_model_tocompiled(base_model, num_class):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_class, activation='softmax', name='predictions')(x)
    clf = Model(base_model.input, predictions)
    return clf



def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--arch', type=str, required=True, \
            choices=["VGG16", "VGG19", "RESNET50", "INCEPTIONV3", "DENSENET121", "MOBILENET", "XCEPTION"], \
            help='architecture (options: VGG16, VGG19, RESNET50, INCEPTIONV3, DENSENET121, MOBILENET, XCEPTION)')

    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=3, type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=31, type=int, help='windows size')
    parser.add_argument('--lrate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.add_argument('--verbosetrain', action='store_true', help='Verbose train')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters: args = set_params(args)

    pixels, labels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    pixels, labels = mydata.createImageCubes(pixels, labels, windowSize=args.spatialsize, removeZeroLabels = False)
    pixels = np.array([cv2.resize(a, (a.shape[0]+1,a.shape[1]+1), interpolation=cv2.INTER_CUBIC)  for a in pixels])

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

        base_model, modlrate = get_model_pretrain(args.arch)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(np.unique(labels)), activation='softmax', name='predictions')(x)
        clf = Model(base_model.input, predictions)

        valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (x_test, keras_to_categorical(y_test, num_class))
        #for layer in clf.layers: layer.trainable = True
        clf.compile(loss=categorical_crossentropy, optimizer=Adam(lr=args.lrate), metrics=['accuracy'])
        clf.fit(x_train, keras_to_categorical(y_train, num_class),
                        batch_size=args.batch_size,
                        epochs=5,
                        verbose=args.verbosetrain,
                        validation_data=valdata,
                        callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_loss', verbose=0, save_best_only=True)])
        del clf; K.clear_session(); gc.collect()

        clf = load_model("/tmp/best_model.h5")
        for layer in base_model.layers: layer.trainable = False
        #for layer in clf.layers[:-3]: layer.trainable = False
        #for idlayer in [-1,-2,-3]: clf.layers[idlayer].trainable = True
        clf.compile(loss=categorical_crossentropy, optimizer=Adam(lr=args.lrate*modlrate), metrics=['accuracy'])
        clf.fit(x_train, keras_to_categorical(y_train, num_class),
                        batch_size=args.batch_size,
                        epochs=50,
                        verbose=args.verbosetrain,
                        validation_data=valdata,
                        callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
        del clf; K.clear_session(); gc.collect()
        clf = load_model("/tmp/best_model.h5")
        #print("PARAMETERS", clf.count_params())
        stats[pos,:] = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print(args.dataset, args.arch, args.tr_percent, list(stats[-1]))

if __name__ == '__main__':
    main()

