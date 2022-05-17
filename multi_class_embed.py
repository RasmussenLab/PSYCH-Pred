import os, sys
import torch
import numpy as np
from torch.utils import data
import re
import random
import copy

from scipy.stats.stats import pearsonr
import scipy
from scipy import stats
import pandas as pd
import seaborn as sns
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data.dataset import TensorDataset

from itertools import combinations
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, accuracy_score
from scipy import interp
import pickle
from itertools import cycle
from sklearn.model_selection import StratifiedKFold

class ClassifierDataset(Dataset):
    
    def __init__(self, emb_data, cat_data, con_data, y_data):
        self.emb_data = emb_data
        self.cat_data = cat_data
        self.con_data = con_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.emb_data[index], self.cat_data[index], self.con_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.cat_data)


class MulticlassClassification(nn.Module):
    def __init__(self, num_cat_feature, num_con_feature, emb_dims, num_class, num_hidden1, num_hidden2):
        super(MulticlassClassification, self).__init__()
        
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y, padding_idx=0)
                                     for x, y in emb_dims])
        
        self.num_embs = sum([y for x, y in emb_dims])
        self.num_con_feature = num_con_feature
        self.num_cat_feature = num_cat_feature
        
        self.layer_1 = nn.Linear(self.num_embs + self.num_cat_feature + self.num_con_feature, num_hidden1)
        self.layer_2 = nn.Linear(num_hidden1, num_hidden2)
        #self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(num_hidden2, num_class) 
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm_cat = nn.BatchNorm1d(self.num_cat_feature)
        self.batchnorm_con = nn.BatchNorm1d(self.num_con_feature)
        self.batchnorm1 = nn.BatchNorm1d(num_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden2)
        
    def forward(self, emb_data, cat_data, con_data):
        
        x = [emb_layer(emb_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        
        nom_cat_data = self.batchnorm_cat(cat_data)
        x = torch.cat([x, nom_cat_data], 1)
        
        nom_con_data = self.batchnorm_con(con_data)
        x = torch.cat([x, nom_con_data], 1)
        
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

def train_model(train_dataset, best_hyp, NUM_FEATURES, NUM_CLASSES, device, best_epochs, labels, emb_dims):
    num_hidden1 = best_hyp[2][0]
    num_hidden2 = best_hyp[2][1]
    BATCH_SIZE = best_hyp[0]
    LEARNING_RATE = best_hyp[1]
    
    # weighted sampler
    target_list = []
    for _, _, _, t in train_dataset:
        target_list.append(t)
    
    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]
    class_count = np.bincount(labels.astype(int))
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)
    
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=weighted_sampler, drop_last = True)
    
    model = MulticlassClassification(num_cat_feature=NUM_FEATURES[0],
                                     num_con_feature = NUM_FEATURES[1],
                                     emb_dims=emb_dims, num_class=NUM_CLASSES,
                                     num_hidden1=num_hidden1, num_hidden2=num_hidden2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    accuracy_stats = []
    loss_stats = []
    
    for e in range(1, best_epochs):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for emb_train_batch, cat_train_batch, con_train_batch, y_train_batch in train_loader:
            emb_train_batch = emb_train_batch.to(device)
            cat_train_batch = cat_train_batch.to(device)
            con_train_batch = con_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)
            
            y_train_pred = model(emb_train_batch, cat_train_batch, con_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc, softmax_pred, correct_pred = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        
        
        loss_stats.append(train_epoch_loss/len(train_loader))
        accuracy_stats.append(train_epoch_acc/len(train_loader))
        
        print('Epoch ' + str(e) + ' | Train Loss: ' + str(train_epoch_loss/len(train_loader)) +
              ' | Train Acc: ' + str(train_epoch_acc/len(train_loader)))
    
    return model, loss_stats, accuracy_stats

def best_hyp_test(best_hyps):
    final_hyps = defaultdict(list)
    trans = dict()
    for bh in best_hyps:
        for i in range(len(bh)):
            final_hyps[i].append(str(bh[i]))
            trans[str(bh[i])] = bh[i]
    
    final_hyp = []
    for fh in final_hyps:
        m_obs = max(set(final_hyps[fh]), key = final_hyps[fh].count)
        final_hyp.append(trans[m_obs])
    
    return final_hyp

def get_w_sampler(train_dataset, labels):
    target_list = []
    for _,_,_, t in train_dataset:
        target_list.append(t)
    
    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]
    class_count = np.bincount(labels.astype(int))
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)
    
    return weighted_sampler, class_weights

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc, y_pred_softmax, y_pred_tags

def evaluate_model_nn(predictions, probs, test_labels, labels_name, f_name=None, colors=None):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions, average='micro')
    results['precision'] = precision_score(test_labels, predictions, average='micro')
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    probs_list = []
    mcc = dict()
    n_classes = probs.shape[1]
    y = label_binarize(test_labels, classes=np.unique(test_labels))
    if y.shape != probs.shape:
        enc = OneHotEncoder()
        y = enc.fit_transform(test_labels.reshape(-1,1)).toarray()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        results[i] = roc_auc[i]
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), np.array(probs).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    results['roc'] =  roc_auc["micro"]
    
    if f_name != None:
        sns.set_style(style='white')
        fig, ax = plt.subplots(figsize = (12, 10))
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['font.size'] = 14
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        ax.grid(b=None, which='both')
        ax.yaxis.grid(b=None, which='both')
        ax.xaxis.grid(b=None, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig(f_name + '.pdf', format = 'pdf', dpi = 1200)
    
    # Plot test across classes
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    if f_name != None:
        # Plot all ROC curves
        sns.set_style(style='white')
        fig, ax = plt.subplots(figsize = (12, 10))
        plt.style.use('seaborn-whitegrid')
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        
        #s_names = np.sort(labels_name)
        s_names = labels_name
        for n, color in zip(s_names, colors):
            i = list(labels_name).index(n)
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(n, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        ax.grid(b=None, which='both')
        ax.yaxis.grid(b=None, which='both')
        ax.xaxis.grid(b=None, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig(f_name + '_all.pdf', format = 'pdf', dpi = 1200)
    
    return results

def train_e(train_dataset, BATCH_SIZE, weighted_sampler, class_weights, NUM_FEATURES, NUM_CLASSES, num_hidden1, num_hidden2, EPOCHS, LEARNING_RATE, val_dataset, val_loss_min, best_hyps_tmp, epoch, labels, device, emb_dims, batch_up, loss_min=True):
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,sampler=weighted_sampler, drop_last = True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=1)
    model = MulticlassClassification(num_cat_feature=NUM_FEATURES[0], num_con_feature = NUM_FEATURES[1],
                                     emb_dims=emb_dims, num_class=NUM_CLASSES, num_hidden1=num_hidden1,
                                     num_hidden2=num_hidden2)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_model = copy.deepcopy(model)
    for e in range(1, EPOCHS+1):
        
        if e in batch_up:
            train_loader = DataLoader(dataset=train_dataset,batch_size=int(BATCH_SIZE*1.5),
                                      sampler=weighted_sampler, drop_last = True)
        
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for emb_train_batch, cat_train_batch, con_train_batch, y_train_batch in train_loader:
            emb_train_batch = emb_train_batch.to(device)
            cat_train_batch = cat_train_batch.to(device)
            con_train_batch = con_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(emb_train_batch, cat_train_batch, con_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc, softmax_pred, correct_pred = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION    
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for emb_val_batch, cat_val_batch, con_val_batch, y_val_batch in val_loader:
                emb_val_batch = emb_val_batch.to(device)
                cat_val_batch = cat_val_batch.to(device)
                con_val_batch = con_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                y_val_pred = model(emb_val_batch, cat_val_batch, con_val_batch)       
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        
        if loss_min == True:
            if val_loss_min > (val_epoch_loss/len(val_loader)):
                val_loss_min = val_epoch_loss/len(val_loader)
                best_hyps_tmp = [BATCH_SIZE, LEARNING_RATE, [num_hidden1,num_hidden2]]
                epoch = e
                best_model = copy.deepcopy(model)
        else:
            if val_loss_min < (val_epoch_acc/len(val_loader)):
                val_loss_min = val_epoch_acc/len(val_loader)
                best_hyps_tmp = [BATCH_SIZE, LEARNING_RATE, [num_hidden1,num_hidden2]]
                epoch = e
                best_model = copy.deepcopy(model)
        print('Epoch ' + str(e) + ' | Train Loss: ' + str(train_epoch_loss/len(train_loader)) +
              ' | Val Loss: ' + str(val_epoch_loss/len(val_loader)) +
              ' | Train Acc: ' + str(train_epoch_acc/len(train_loader))
              + ' | Val Acc: ' + str(val_epoch_acc/len(val_loader)))
    
    return val_loss_min, best_hyps_tmp, epoch, best_model

def make_conf_matrix(test_labels, predictions, labels_name_not_sort, labels_name, f_name):
    cmap = sns.diverging_palette(220, 20, sep=10, as_cmap=True)
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    conf_mat = confusion_matrix(test_labels, np.array(predictions))
    labels_name_order = pd.Categorical(labels_name, categories = labels_name, ordered = True)
    conf_mat = pd.DataFrame(conf_mat, labels_name_not_sort, labels_name_not_sort)
    conf_mat = conf_mat.loc[labels_name_order,labels_name_order]
    fig = plt.figure(figsize=(14,10))
    g = sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt = 'd', center = 0) # font size
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=0)
    g.set_yticklabels(g.yaxis.get_majorticklabels(), rotation=0)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f_name, format = 'pdf', dpi = 1200)

def make_conf_matrix_frac(test_labels, predictions, labels_name_not_sort, labels_name, f_name):
    cmap = sns.diverging_palette(20, 220, sep=5, as_cmap=True)
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    conf_mat = confusion_matrix(test_labels, np.array(predictions))
    labels_name_order = pd.Categorical(labels_name, categories = labels_name, ordered = True)
    conf_mat = pd.DataFrame(conf_mat, labels_name_not_sort, labels_name_not_sort)
    conf_mat = conf_mat.astype(int)
    conf_mat = conf_mat.loc[labels_name_order,labels_name_order]
    
    conf_mat_frac = confusion_matrix(test_labels, np.array(predictions), normalize='true')
    conf_mat_frac = pd.DataFrame(conf_mat_frac, labels_name_not_sort, labels_name_not_sort)
    conf_mat_frac = conf_mat_frac.loc[labels_name_order,labels_name_order]
    
    fig = plt.figure(figsize=(14,10))
    g = sns.heatmap(conf_mat_frac, annot=conf_mat, annot_kws={"size": 16}, cmap=cmap, fmt = 'd', center = 0) # font size
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=0)
    g.set_yticklabels(g.yaxis.get_majorticklabels(), rotation=0)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f_name, format = 'pdf', dpi = 1200)

def test_model(best_model, test_dataset, device, labels):
    weighted_sampler, class_weights = get_w_sampler(test_dataset, labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_acc = 0
        predictions = []
        probs = []
        
        best_model.eval()
        for emb_val_batch, cat_val_batch, con_val_batch, y_val_batch in test_loader:
            emb_val_batch = emb_val_batch.to(device)
            cat_val_batch = cat_val_batch.to(device)
            con_val_batch = con_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            
            y_val_pred = best_model(emb_val_batch, cat_val_batch, con_val_batch)
                        
            test_loss = criterion(y_val_pred, y_val_batch)
            test_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            test_epoch_loss += test_loss.item()
            test_epoch_acc += test_acc.item()
            predictions.append(int(correct_pred))
            probs.append(np.array(y_val_pred).ravel())
    
    test_loss = test_epoch_loss / len(test_loader)
    test_acc = test_epoch_acc / len(test_loader)
    return test_acc, test_loss, predictions, probs
