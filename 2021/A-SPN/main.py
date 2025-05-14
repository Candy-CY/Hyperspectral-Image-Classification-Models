# -*- coding: utf-8 -*-
"""
@author: mengxue.zhang
"""

import os
from functions import loop_train_test
from data import image_size_dict as dims
from data import draw_false_color, draw_gt, draw_bar

# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## global constants
verbose = 1 # whether or not print redundant info (1 if and only if in debug mode, 0 in run mode)
run_times = 1 # random run times, recommend at least 10
output_map = False # whether or not output classification map
only_draw_label = False # whether or not only predict labeled samples
disjoint = False # whether or not train and test on spatially disjoint samples

lr = 1e-2 # init learing rate
decay = 1e-2 # exponential learning rate decay
ws = 9 # window size
epochs = 15 # epoch
batch_size = 64 # batch size
model_type = 'aspn' # model type in {'spn', 'aspn'}

def indian_pine_experiment():
    hp = {
        'pc': dims['2'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 9]
    loop_train_test(dataID=2, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)

def pavia_university_experiment():
    hp = {
        'pc': dims['1'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [332, 932, 105, 153, 67, 251, 67, 184, 47]
    loop_train_test(dataID=1, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)

def houston_university_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [50] * 15
    loop_train_test(dataID=3, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


indian_pine_experiment()
pavia_university_experiment()
houston_university_experiment()

# draw_false_color(dataID=1)
# draw_false_color(dataID=2)
# draw_false_color(dataID=3)
#
# draw_bar(dataID=1)
# draw_bar(dataID=2)
# draw_bar(dataID=3)

# draw_gt(dataID=1, fixed=disjoint)
# draw_gt(dataID=2, fixed=disjoint)
# draw_gt(dataID=3, fixed=disjoint)

