#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 03:28:40 2023

@author: Rojan Basnet
"""

# ALL LIBRARIES
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import models
import data_preprocess

# Hyper Parameters
# Define the hyperparameters for the experiment
feature_dimension = 160
source_input_dimension = 128
target_input_dimension = 103
n_dimension = 100
num_classes = 9
shots_per_class = 1
queries_per_class = 19
training_episodes = 20000
test_episodes = 600
learning_rate = 0.001
gpu_index = 0
hidden_units = 10

# Hyper Parameters in target domain data set
test_num_classes = 9  # the number of classes
test_labeled_samples_per_class = 5  # the number of labeled samples per class

# Set random seeds
utils.set_random_seeds(0)

# Initialize directories for saving checkpoints and classification maps
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# Load source domain dataset
with open(os.path.join('datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print("Source domain dataset keys:", source_imdb.keys())
print("Source domain dataset labels:", source_imdb['Labels'])

# Process source domain dataset
data_train = source_imdb['data']
labels_train = source_imdb['Labels']
print("Source domain dataset shape:", data_train.shape)
print("Source domain dataset label shape:", labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))
print("All unique labels in source domain dataset:", keys_all_train)
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print("Label encoder for source domain dataset:", label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print("Classes in the source domain dataset:", train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Number of classes for source domain dataset:", len(data))
print("Classes in the source domain dataset after sanity check:", data.keys())
data = utils.filter_valid_classes(data)
print("Number of classes with more than 200 samples:", len(data))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# Source few-shot classification data
metatrain_data = data
print("Number of classes in source few-shot classification data:", len(metatrain_data.keys()))
del data

# Source domain adaptation data
print("Source domain data shape before transpose:", source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print("Source domain data shape after transpose:", source_imdb['data'].shape)
print("Source domain dataset labels:", source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## Target domain dataset
# Load target domain dataset
test_data = 'datasets/paviaU/paviaU.mat'
test_label = 'datasets/paviaU/paviaU_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_image_data(test_data, test_label)

# Run the experiment multiple times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, num_classes])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # Load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = data_preprocess.get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=test_num_classes, shot_num_per_class=test_labeled_samples_per_class)
    
    # Model
    feature_encoder = models.Network(feature_dimension, num_classes, target_input_dimension, source_input_dimension, n_dimension, 1, 1)
    domain_classifier = models.DomainClassifier()
    random_layer = models.RandomLayer([feature_dimension, num_classes], 1024) 

    feature_encoder.apply(models.weights_init)
    domain_classifier.apply(models.weights_init)

    feature_encoder.cuda()
    domain_classifier.cuda()
    random_layer.cuda()

    feature_encoder.train()
    domain_classifier.train()
    
    # Optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    
    for episode in range(training_episodes):
        # Get domain adaptation data from source domain and target domain
        try:
            source_data, source_label = source_iter.__next__()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.__next__()

        try:
            target_data, target_label = target_iter.__next__()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.__next__()

        # Source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot classification for source domain dataset'''
            # Get few-shot classification samples
            task = utils.Task(metatrain_data, num_classes, shots_per_class, queries_per_class)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=shots_per_class, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=queries_per_class, split="test", shuffle=True)

            # Sample data
            supports, support_labels = support_dataloader.__iter__().__next__()
            querys, query_labels = query_dataloader.__iter__().__next__()

            # Calculate features
            support_features, support_outputs = feature_encoder(supports.cuda())
            query_features, query_outputs = feature_encoder(querys.cuda())
            target_features, target_outputs = feature_encoder(target_data.cuda(), domain='target')

            # Prototype network
            if shots_per_class > 1:
                support_proto = support_features.reshape(num_classes, shots_per_class, -1).mean(dim=1)
            else:
                support_proto = support_features

            # FSL loss
            logits = models.pairwise_euclidean_distance(query_features, support_proto)
            f_loss = models.crossEntropy(logits, query_labels.cuda())

            '''Domain adaptation'''
            # Calculate domain adaptation loss
            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            # Set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label[:supports.shape[0] + querys.shape[0]] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = models.domain_criterion(domain_logits, domain_label)

            # Total loss = FSL loss + domain loss
            loss = f_loss + domain_loss

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        
        # Target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain dataset'''
            # Get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, test_num_classes, shots_per_class, queries_per_class)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=shots_per_class, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=queries_per_class, split="test", shuffle=True)

            # Sample data
            supports, support_labels = support_dataloader.__iter__().__next__()
            querys, query_labels = query_dataloader.__iter__().__next__()

            # Calculate features
            support_features, support_outputs = feature_encoder(supports.cuda(), domain='target')
            query_features, query_outputs = feature_encoder(querys.cuda(), domain='target')
            source_features, source_outputs = feature_encoder(source_data.cuda())

            # Prototype network
            if shots_per_class > 1:
                support_proto = support_features.reshape(num_classes, shots_per_class, -1).mean(dim=1)
            else:
                support_proto = support_features

            # FSL loss
            logits = models.pairwise_euclidean_distance(query_features, support_proto)
            f_loss = models.crossEntropy(logits, query_labels.cuda())

            '''Domain adaptation'''
            features = torch.cat([support_features, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[supports.shape[0] + querys.shape[0]:] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = models.domain_criterion(domain_logits, domain_label)

            # Total loss = FSL loss + domain loss
            loss = f_loss + domain_loss

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # Display progress every 100 episodes
            train_loss.append(loss.item())
            print('Episode {:>3d}:  Domain loss: {:6.4f}, FSL loss: {:6.4f}, Accuracy: {:6.4f}, Total loss: {:6.4f}'.format(
                episode + 1, domain_loss.item(), f_loss.item(), total_hit / total_num, loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # Test the model
            print("Testing...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            # Calculate features for training set
            train_datas, train_labels = train_loader.__iter__().__next__()
            train_features, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')

            # Normalize features
            max_value = train_features.max()
            min_value = train_features.min()
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            # Fit KNN classifier
            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            
            # Test on test set
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                # Calculate features for test set
                test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset), test_accuracy))
            test_end = time.time()

            # Set the model back to training mode
            feature_encoder.train()

            if test_accuracy > last_accuracy:
                # Save the model checkpoints
                torch.save(feature_encoder.state_dict(),
                           str("checkpoints/DFSL_feature_encoder_" + "salinas_" + str(iDataSet) + "iter_" + str(
                               test_labeled_samples_per_class) + "shot.pkl"))
                print("Saved networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('Best episode: [{}], Best accuracy: {}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('Iter: {} Best episode: [{}], Best accuracy: {}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)

print("Train time per DataSet (s): {:.5f}".format(train_end - train_start))
print("Test time per DataSet (s): {:.5f}".format(test_end - train_end))
print("Average OA: {:.2f} +- {:.2f}".format(OAMean, OAStd))
print("Average AA: {:.2f} +- {:.2f}".format(100 * AAMean, 100 * AAStd))
print("Average kappa: {:.4f} +- {:.4f}".format(100 * kMean, 100 * kStd))
print("Accuracy for each class:")
for i in range(num_classes):
    print("Class {}: {:.2f} +- {:.2f}".format(i, 100 * AMean[i], 100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}: {}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('Best accuracy for all: {}'.format(acc[best_iDataset]))

################# Classification Map ############################
# Convert predictions to classification map
classification_map = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(len(best_predict_all)):
    index = best_nTrain + i
    row_index = best_Row[best_RandPerm[index]]
    col_index = best_Column[best_RandPerm[index]]
    classification_map[row_index][col_index] = best_predict_all[i] + 1

# Map classification values to colors
colors = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [0, 1, 1],
    4: [1, 0, 0],
    5: [1, 0, 1],
    6: [1, 1, 0],
    7: [0.5, 0.5, 1],
    8: [0.65, 0.35, 1],
    9: [0.75, 0.5, 0.75],
    10: [0.75, 1, 0.5],
    11: [0.5, 1, 0.65],
    12: [0.65, 0.65, 0],
    13: [0.75, 1, 0.65],
    14: [0, 0, 0.5],
    15: [0, 1, 0.75],
    16: [0.5, 0.75, 1]
}

# Generate the final classification map
classification_map_rgb = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        classification_value = best_G[i][j]
        if classification_value in colors:
            classification_map_rgb[i, j, :] = colors[classification_value]

# Generate and save the classification map image
classification_map_cropped = classification_map_rgb[4:-4, 4:-4, :]
output_file_path = "classificationMap/salinas_{}shot.png".format(test_labeled_samples_per_class)
utils.classification_map(classification_map_cropped, classification_map[4:-4, 4:-4], 24, output_file_path)