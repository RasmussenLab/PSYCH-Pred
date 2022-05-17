from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import power_transform
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances
from scipy import stats
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import random
import copy
import scipy
from scipy import stats
plt.style.use('seaborn-whitegrid')

import os, sys
import torch
import numpy as np
from torch.utils import data
import re

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data.dataset import TensorDataset

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from scipy import interp
import pickle
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer

## Functions

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


path = 'home'
sys.path.append(path + "/prediction/scripts/")
from plots import embedding_plot_discrete, embedding_plot_float, plot_error
from read_files import encode_binary, remove_not_obs_cat, encode_con, read_cat, read_con, read_header, concat_con_list, concat_cat_list
from multi_class_emb import train_model, best_hyp_test, get_w_sampler, multi_acc, evaluate_model_nn, train_e, make_conf_matrix, test_model, test_model, make_conf_matrix_frac

p = 0.001
labels_names = np.load(path + "/data_encoded/included_patient_labels.npy")

## load in continuous pheno data and feature names/header(encoded and raw/age values)
F_pheno = read_con(path + "/data_encoded/input/pheno_F_con.npy")
F_pheno, mask = encode_con(F_pheno, p)
F_pheno_h = read_header(path + "/data_encoded/phenotypes_age/pheno_F_headers_con.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/pheno_F_con.npy")
con_all_raw = tmp_raw[:,mask]

severity_pheno = read_con(path + "/data_encoded/input/sev_con.npy")
severity_pheno, mask = encode_con(severity_pheno, p)
severity_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sev_con_headers.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/sev_con.npy")
tmp_raw = tmp_raw[:,mask]
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

mbr = read_con(path + "/data_encoded/input/mbr_con_age.npy")
mbr, mask = encode_con(mbr, p)
mbr_h = read_header(path + "/data_encoded/phenotypes_age/mbr_con_headers_age.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/mbr_con_age.npy")
tmp_raw = tmp_raw[:,mask]
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

LPR = read_con(path + "/data_encoded/input/other_LPR_con.npy")
LPR, mask = encode_con(LPR, p)
LPR_h = read_header(path + "/data_encoded/phenotypes_age/other_LPR_headers_con.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/other_LPR_con.npy")
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)

PRS = read_con(path + "/data_encoded/input/PRS_con_v3.npy")
PRS, mask = encode_con(PRS, p)
PRS_h = read_header(path + "/data_encoded/PRS/PRS_header_v3.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/PRS_con_v3.npy")
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)

## load pheno data categorical
MBR_pheno, MBR_pheno_input = read_cat(path + "/data_encoded/input/mbr_cat_age.npy")
MBR_pheno_h = read_header(path + "/data_encoded/phenotypes_age/mbr_cat_headers_age.txt")
MBR_pheno, MBR_pheno_input, MBR_pheno_h = remove_not_obs_cat(MBR_pheno, MBR_pheno_input, MBR_pheno_h, p)

sibling_pheno, sibling_pheno_input = read_cat(path + "/data_encoded/input/sibling_cat.npy")
sibling_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sibling_cat_headers.txt")
sibling_pheno, sibling_pheno_input, sibling_pheno_h = remove_not_obs_cat(sibling_pheno, sibling_pheno_input, sibling_pheno_h, p)

## load in genotype and HLA
hla_pheno, hla_pheno_input = read_cat(path + "/data_encoded/input/geno_hla.npy")
hla_pheno_h = read_header(path + "/data_encoded/genomics/geno_hla_headers.txt")
hla_pheno, hla_pheno_input, hla_pheno_h = remove_not_obs_cat(hla_pheno, hla_pheno_input, hla_pheno_h, p)

geno, geno_input = read_cat(path + "/data_encoded/input/genotypes_all.npy")
geno_h = read_header(path + "/data_encoded/genomics/genotypes_headers_all.txt")
geno, geno_input, geno_h = remove_not_obs_cat(geno, geno_input, geno_h, p)

# Load binary LPR diagnosis
f_LPR = read_con(path + "/data_encoded/input/father_LPR_con.npy")
f_LPR, f_LPR_input, mask = encode_binary(f_LPR, p)
f_LPR_h = read_header(path + "/data_encoded/phenotypes_age/father_LPR_headers_con.txt", mask)

m_LPR = read_con(path + "/data_encoded/input/mother_LPR_con.npy")
m_LPR, m_LPR_input, mask = encode_binary(m_LPR, p)
m_LPR_h = read_header(path + "/data_encoded/phenotypes_age/mother_LPR_headers_con.txt", mask)

# combine parents and sibling
family_LPR = np.concatenate((f_LPR, m_LPR, sibling_pheno), axis=1)
family_LPR_h = np.concatenate((f_LPR_h, m_LPR_h, sibling_pheno_h))
family_LPR_input = np.concatenate((f_LPR_input, m_LPR_input, sibling_pheno_input), axis=1)

## Set paracombinemeters
analysis_type = "all_diagnoses"
version = 'v1'

sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')

first_age = np.load(path + "/data_encoded/included_patient_first_age.npy")
first_age = first_age[labels_names != '']
first_age = first_age[np.sum(geno_input, axis=1) != 0]

ages = np.load('/faststorage/jail/project/gentofte_projects/prediction/data_encoded/included_patient_age.npy')
ages = ages[labels_names != '']
ages = ages[np.sum(geno_input, axis=1) != 0]

labels_names = np.load(path + "/data_encoded/included_patient_labels.npy")
y = label_binarize(labels_names, classes=np.unique(labels_names))
labels = np.argmax(y, 1)

n_classes = y.shape[1]

labels_name_not_sort = list(np.unique(labels_names))
labels_name = ['Back_pop', 'ADHD', 'ASD', 'MDD', 'BD', 'SCZ']

# Prepare data
mbr_geno = np.concatenate((MBR_pheno, hla_pheno, geno), axis=1)
mbr_geno_h = np.concatenate((MBR_pheno_h, hla_pheno_h, geno_h))
mbr_geno_input = np.concatenate((MBR_pheno_input, hla_pheno_input, geno_input), axis=1)

cat_names = family_LPR_h
emb_names = np.concatenate((MBR_pheno_h, hla_pheno_h, geno_h))
con_names = np.concatenate((F_pheno_h, severity_pheno_h, mbr_h, LPR_h, PRS_h))

all_data = np.concatenate((F_pheno, severity_pheno, mbr, LPR, PRS, family_LPR_input,mbr_geno_input), axis=1)
data_df = pd.DataFrame(all_data, columns = np.concatenate((con_names, cat_names, emb_names)))

emb_list = [MBR_pheno_input, hla_pheno_input, geno_input]
emb_shapes, mask, emb_all = concat_cat_list(emb_list)

cat_list = [family_LPR_input]
cat_shapes, mask, cat_all = concat_cat_list(cat_list)

con_list = [F_pheno, severity_pheno, mbr, LPR, PRS]
n_con_shapes, mask, con_all = concat_con_list(con_list, mask)

all_raw = np.concatenate((con_all_raw, family_LPR_input, mbr_geno_input), axis=1)
data_df_raw = pd.DataFrame(all_raw, columns = np.concatenate((con_names,cat_names, emb_names)))

# Remove all observations from diagnosis
filtered = []
tmp_data = data_df_raw[np.concatenate((F_pheno_h, LPR_h))]
diag_age = np.load(path + 'data/included_patient_age_corr_all.npy')
for j in range(len(first_age)):
    indi_data = tmp_data.loc[j,:]
    indi_data[indi_data >= new_age[j]] = 0
    filtered.append(np.array(ep))

new_age_enc, mask = encode_con(np.array(new_age)[:, None], p)
filtered = np.array(filtered)

# Make new dataset filtered by age
p = 0.0001
F_pheno_h_new = F_pheno_h
filtered_enc, mask_enc = encode_con(filtered, p)
filtered_h = np.concatenate((F_pheno_h_new, LPR_h))[mask_enc]
data_filtered_h = np.concatenate((filtered_h, ['age'], PRS_h, mbr_h, family_LPR_h, mbr_geno_h))
data_filtered = np.column_stack((filtered_enc, new_age_enc, PRS, mbr, family_LPR_input, mbr_geno_input))
data_df_filtered = pd.DataFrame(data_filtered, columns = data_filtered_h)

emb_list = [MBR_pheno_input, hla_pheno_input, geno_input]
emb_shapes, mask, emb_all = concat_cat_list(emb_list)

cat_list = [family_LPR_input]
cat_shapes, mask, cat_all = concat_cat_list(cat_list)

con_names = np.concatenate((filtered_h, ['age'], PRS_h, mbr_h))
con_list = [filtered_enc, new_age_enc, PRS, mbr]
n_con_shapes, mask, con_all = concat_con_list(con_list, mask)

RSEED = 42
skf_test = StratifiedKFold(n_splits=3, random_state = RSEED, shuffle=True)

EPOCHS = 25
NUM_FEATURES = [len(cat_names), len(con_names)]
cat_dims = [int(np.max(data_df.loc[:,col])) + 1 for col in emb_names]

emb_dims = [(x, min(50,round((1.6 * x ** 0.56)))) for x in cat_dims]

NUM_CLASSES = len(np.unique(labels))
cuda = False
device = torch.device("cuda" if cuda == True else "cpu")
colors = cycle(['lightskyblue','royalblue', 'darkblue', 'salmon', 'red', 'crimson', 'maroon'])

# Hyper parameters
val_loss_min = 0
batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.0001, 0.00001]
num_hiddens = [[64, 28], [128, 64], [256, 128]]

performance = []
best_hyps = []
test_predictions = []
test_probs = []
test_idx = []
models = []
nr = 1

batch_up = [5, 10, 20]
data_df = data_df_filtered
for train_index_1, test_index in skf_test.split(data_df, labels):
    test_idx.append(test_index)
    train_1, test = data_df.loc[train_index_1, :], data_df.loc[test_index, :]
    train_labels_1, test_labels = labels[train_index_1], labels[test_index]
    
    emb_test, cat_test, con_test = test.loc[:,emb_names], test.loc[:,cat_names], test.loc[:,con_names]
    test_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_test)).long(),
                                     torch.from_numpy(np.array(cat_test)).float(),
                                     torch.from_numpy(np.array(con_test)).float(),
                                     torch.from_numpy(test_labels).long())
    
    best_hyps_val = []
    epochs = []
    train, val, train_labels, val_labels = train_test_split(train_1, train_labels_1, 
                                                          stratify = train_labels_1,
                                                          test_size = 0.1, random_state = RSEED)
    
    emb_train, cat_train, con_train= train.loc[:,emb_names], train.loc[:,cat_names], train.loc[:,con_names]
    train_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_train)).long(),
                                      torch.from_numpy(np.array(cat_train)).float(),
                                      torch.from_numpy(np.array(con_train)).float(),
                                      torch.from_numpy(train_labels).long())
    
    emb_val, cat_val, con_val = val.loc[:, emb_names], val.loc[:,cat_names], val.loc[:,con_names]
    val_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_val)).long(),
                                    torch.from_numpy(np.array(cat_val)).float(),
                                    torch.from_numpy(np.array(con_val)).float(),
                                    torch.from_numpy(val_labels).long())
    
    # weighted sampler
    weighted_sampler, class_weights = get_w_sampler(train_dataset, labels)
    criterion = nn.CrossEntropyLoss()
    
    val_loss_min = 0
    old_min = 0
    best_hyps_tmp = [batch_sizes[0], learning_rates[0], num_hiddens[0]]
    epoch = np.copy(EPOCHS)
    for BATCH_SIZE in batch_sizes:
        for LEARNING_RATE in learning_rates:
            for num_hidden in num_hiddens:
                num_hidden1 = num_hidden[0]
                num_hidden2 = num_hidden[1]
                
                val_loss_min, best_hyps_tmp, epoch, model = train_e(train_dataset, BATCH_SIZE,
                                                                    weighted_sampler, class_weights,
                                                                    NUM_FEATURES, NUM_CLASSES,
                                                                    num_hidden1, num_hidden2,
                                                                    EPOCHS, LEARNING_RATE,
                                                                    val_dataset, val_loss_min,
                                                                    best_hyps_tmp, epoch, labels,
                                                                    device, emb_dims, batch_up,
                                                                    loss_min=False)
                
                if val_loss_min > old_min:
                    best_model = copy.deepcopy(model)
                    old_min = val_loss_min
            epochs.append(epoch)
    
    best_epochs = max(set(epochs), key = epochs.count)
    test_acc, test_loss, predictions, probs = test_model_new(best_model, test_dataset, device, labels)
    test_predictions.append(predictions)
    test_probs.append(probs)
    f_name = path + "/prediction/" + analysis_type + "/nn_roc_" + version  + "_" + analysis_type + str(nr)
    test_eval = evaluate_model_nn(np.array(predictions), np.array(probs), test_labels, labels_name, f_name, colors)
    mcc_all = matthews_corrcoef(test_labels, np.array(predictions))
    f_name = path + "/prediction/" + analysis_type + "/nn_confusion_matrix_" + version  + "_" + analysis_type + str(nr) + '.pdf'
    make_conf_matrix(test_labels, predictions, labels_name_not_sort, labels_name, f_name)
    performance.append([test_eval, mcc_all, test_acc])
    best_hyps.append(best_hyps_tmp)
    models.append(best_model)
    nr += 1
  
# Best model
for j in range(len(models)):
    filename = path + "/prediction/" + analysis_type + "/nn_model_filtered_" + str(j) + "_" + version + "_" + analysis_type + ".pt"
    torch.save(models[j], filename)

np.save(path + "/prediction/" + analysis_type + "/nn_predictions_filtered_" + version + "_" + analysis_type + ".npy", test_predictions)
np.save(path + "/prediction/" + analysis_type + "/nn_probs_filtered_" + version + "_" + analysis_type + ".npy", test_probs)
np.save(path + "/prediction/" + analysis_type + "/nn_performance_" + version + "_" + analysis_type + ".npy", performance)
np.save(path + "/prediction/" + analysis_type + "/nn_best_hyps_" + version + "_" + analysis_type + ".npy", best_hyps)

# Make collected performancece
predictions_all = np.concatenate(test_predictions)
probs_all = np.concatenate(test_probs)
test_labels_all = np.concatenate([labels[test_idx[0]], labels[test_idx[1]],labels[test_idx[2]]])

# Make plots
f_name = path + "/prediction/" + analysis_type + "/nn_roc_" + version  + "_" + analysis_type
test_eval = evaluate_model_nn(predictions_all, probs_all, test_labels_all, labels_name, f_name, colors)
np.save(test_eval)

mcc_all = matthews_corrcoef(test_labels_all, predictions_all)
f_name = path + "/prediction/" + analysis_type + "/nn_confusion_matrix_" + version  + "_" + analysis_type + '.pdf'
make_conf_matrix(test_labels_all, predictions_all, labels_name_not_sort, labels_name, f_name)
f_name = path + "/prediction/" + analysis_type + "/nn_confusion_matrix_frac_" + version  + "_" + analysis_type + '.pdf'
make_conf_matrix_frac(test_labels_all, predictions_all, labels_name_not_sort, labels_name, f_name)
