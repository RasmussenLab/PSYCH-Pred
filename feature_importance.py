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
from collections import defaultdict

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
        #self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(num_hidden2, num_class) 
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm_cat = nn.BatchNorm1d(self.num_cat_feature)
        self.batchnorm_con = nn.BatchNorm1d(self.num_con_feature)
        self.batchnorm1 = nn.BatchNorm1d(num_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden2)
        #self.batchnorm1 = nn.GroupNorm(1, num_hidden1)
        #self.batchnorm2 = nn.GroupNorm(1, num_hidden2)
        #self.batchnorm3 = nn.BatchNorm1d(32)
        
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
        
        #x = self.layer_3(x)
        #x = self.batchnorm3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

#path = "/data/projects/IBP/rosa/VAE_V1/"
#path = "/data/projects/IBP/to_transfer_to_computerome2/VAE_rosa/"
path = '/faststorage/jail/project/gentofte_projects/'
sys.path.append(path + "prediction/scripts/")
from plots import embedding_plot_discrete, embedding_plot_float, plot_error
from read_files import encode_binary, encode_cat, encode_con, remove_not_obs_cat, remove_not_obs_ordinal, read_cat, read_con, read_header, concat_con_list, concat_cat_list_ordinal, concat_cat_list
from multi_class_emb import train_model, best_hyp_test, get_w_sampler, multi_acc, evaluate_model_nn, train_e, make_conf_matrix, test_model, test_model_new, make_conf_matrix_frac

p = 0.001
#path = "/data/projects/IBP/to_transfer_to_computerome2/VAE_rosa/data/new_prediction/"
path = '/faststorage/jail/project/gentofte_projects/VAE_rosa/data/new_prediction/'
labels_names = np.load(path + "/data_encoded/included_patient_labels.npy")

geno, geno_input = read_cat(path + "/data_encoded/input/genotypes_all.npy")
geno = geno[labels_names != '',:]
geno_input = geno_input[labels_names != '',:]
geno_h = read_header(path + "/data_encoded/genomics/genotypes_headers_all.txt")
geno, geno_input, geno_h = remove_not_obs_ordinal(geno, geno_input, geno_h, p)
geno = geno[np.sum(geno_input, axis=1) != 0]

F_pheno = read_con(path + "/data_encoded/input/pheno_F_con.npy")
F_pheno = F_pheno[labels_names != '',:]
F_pheno = F_pheno[np.sum(geno_input, axis=1) != 0]
F_pheno, mask = encode_con(F_pheno, p)
F_pheno_h = read_header(path + "/data_encoded/phenotypes_age/pheno_F_headers_con.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/pheno_F_con.npy")
con_all_raw = tmp_raw[:,mask]
con_all_raw = con_all_raw[labels_names != '',:]
con_all_raw = con_all_raw[np.sum(geno_input, axis=1) != 0]

tmp_h = [i for i in F_pheno_h if not i.startswith('age_F')]
other_LPR = F_pheno[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
other_LPR_h = F_pheno_h[np.where(np.isin(F_pheno_h,tmp_h))]
other_raw = con_all_raw[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))

tmp_h = [i for i in F_pheno_h if i.startswith('age_F')]
F_pheno =  F_pheno[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
con_all_raw = con_all_raw[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
F_pheno_h = F_pheno_h[np.where(np.isin(F_pheno_h,tmp_h))]

# remove collacted measurements
# h_to_remove = ['age_F1000', 'age_F2001', 'age_F2100', 'age_F2101', 'age_F3000', 'age_F3001', 'age_F3101', 'age_F4101', 'age_F4300', 'age_F5101', 'age_F5200', 'age_F6100', 'age_F7000', 'age_F8100', 'age_F8101', 'age_F8300', 'age_F9100', 'age_F9201', 'age_F9202', 'age_F9203', 'age_F9204','age_F9297']
# tmp_h = [i for i in F_pheno_h if i not in h_to_remove]
# F_pheno =  F_pheno[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
# con_all_raw = con_all_raw[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(con_all_raw.shape[0],len(tmp_h))
# F_pheno_h = F_pheno_h[np.where(np.isin(F_pheno_h,tmp_h))]

## load in continuous pheno data
severity_pheno = read_con(path + "/data_encoded/input/sev_con.npy")
severity_pheno = severity_pheno[labels_names != '',:]
severity_pheno = severity_pheno[np.sum(geno_input, axis=1) != 0]
severity_pheno, mask = encode_con(severity_pheno, p)
severity_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sev_con_headers.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/sev_con.npy")
tmp_raw = tmp_raw[:,mask]
tmp_raw = tmp_raw[labels_names != '',:]
tmp_raw = tmp_raw[np.sum(geno_input, axis=1) != 0]
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

mbr = read_con(path + "/data_encoded/input/mbr_con_age.npy")
mbr[:,4]  = mbr[:,4] / 1000
mbr = mbr[labels_names != '',:]
mbr = mbr[np.sum(geno_input, axis=1) != 0]
mbr, mask = encode_con(mbr, p)
mbr_h = read_header(path + "/data_encoded/phenotypes_age/mbr_con_headers_age.txt", mask)
mbr_h = np.delete(mbr_h, 3)
mbr = np.delete(mbr, 3, axis=1)

tmp_raw = read_con(path + "/data_encoded/input/mbr_con_age.npy")
tmp_raw[:,4]  = tmp_raw[:,4] / 1000
tmp_raw = tmp_raw[labels_names != '',:]
tmp_raw = tmp_raw[:,mask]
tmp_raw = np.delete(tmp_raw, 3, axis=1)
tmp_raw = tmp_raw[np.sum(geno_input, axis=1) != 0]
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

LPR = read_con(path + "/data_encoded/input/other_LPR_con.npy")
LPR = LPR[labels_names != '',:]
LPR = LPR[np.sum(geno_input, axis=1) != 0]
#LPR, LPR_input, mask = encode_binary(LPR, 0.01)
LPR, mask = encode_con(LPR, p)
LPR_h = read_header(path + "/data_encoded/phenotypes_age/other_LPR_headers_con.txt", mask)
LPR = np.concatenate((LPR, other_LPR), axis=1)
LPR_h = np.concatenate((LPR_h, other_LPR_h))

tmp_raw = read_con(path + "/data_encoded/input/other_LPR_con.npy")
tmp_raw = tmp_raw[labels_names != '',:]
tmp_raw = tmp_raw[np.sum(geno_input, axis=1) != 0]
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)
con_all_raw = np.concatenate((con_all_raw, other_raw), axis=1)

PRS = read_con(path + "/data_encoded/input/PRS_con_v3.npy")
PRS = PRS[labels_names != '',:]
PRS = PRS[np.sum(geno_input, axis=1) != 0]
PRS, mask = encode_con(PRS, p)
PRS_h = read_header(path + "/data_encoded/PRS/PRS_header_v3.txt", mask)
result = set()
for fname in PRS_h:
    orig = fname
    i=1
    while fname in result:
        fname = orig + str(i)
        i += 1
    result.add(fname)

PRS_h = list(result)

tmp_raw = read_con(path + "/data_encoded/input/PRS_con_v3.npy")
tmp_raw = tmp_raw[labels_names != '',:]
tmp_raw = tmp_raw[np.sum(geno_input, axis=1) != 0]
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)

## load pheno data categorical

MBR_pheno, MBR_pheno_input = read_cat(path + "/data_encoded/input/mbr_cat_age.npy")
MBR_pheno = MBR_pheno[labels_names != '',:]
MBR_pheno = MBR_pheno[np.sum(geno_input, axis=1) != 0]
MBR_pheno_input = MBR_pheno_input[labels_names != '',:]
MBR_pheno_input = MBR_pheno_input[np.sum(geno_input, axis=1) != 0]
MBR_pheno_h = read_header(path + "/data_encoded/phenotypes_age/mbr_cat_headers_age.txt")
MBR_pheno, MBR_pheno_input, MBR_pheno_h = remove_not_obs_cat(MBR_pheno, MBR_pheno_input, MBR_pheno_h, p)

sibling_pheno, sibling_pheno_input = read_cat(path + "/data_encoded/input/sibling_cat.npy")
sibling_pheno = sibling_pheno[labels_names != '',:]
sibling_pheno = sibling_pheno[np.sum(geno_input, axis=1) != 0]
sibling_pheno_input = sibling_pheno_input[labels_names != '',:]
sibling_pheno_input = sibling_pheno_input[np.sum(geno_input, axis=1) != 0]
sibling_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sibling_cat_headers.txt")
sibling_pheno, sibling_pheno_input, sibling_pheno_h = remove_not_obs_cat(sibling_pheno, sibling_pheno_input, sibling_pheno_h, p)
sibling_pheno = np.compress((sibling_pheno!=0).sum(axis=(0,1)), sibling_pheno, axis=2)

# combine MBR and sibling
#MBR_sibling = np.concatenate((MBR_pheno, sibling_pheno), axis=1)
#MBR_sibling_h = np.concatenate((MBR_pheno_h, sibling_pheno_h))

## load in genotype
hla_pheno, hla_pheno_input = read_cat(path + "/data_encoded/input/geno_hla.npy")
hla_pheno = hla_pheno[labels_names != '',:]
hla_pheno = hla_pheno[np.sum(geno_input, axis=1) != 0]
hla_pheno_input = hla_pheno_input[labels_names != '',:]
hla_pheno_input = hla_pheno_input[np.sum(geno_input, axis=1) != 0]
hla_pheno_h = read_header(path + "/data_encoded/genomics/geno_hla_headers.txt")
hla_pheno, hla_pheno_input, hla_pheno_h = remove_not_obs_ordinal(hla_pheno, hla_pheno_input, hla_pheno_h, p)

# Load binary LPR diagnosis
f_LPR = read_con(path + "/data_encoded/input/father_LPR_con.npy")
f_LPR = f_LPR[labels_names != '',:]
f_LPR = f_LPR[np.sum(geno_input, axis=1) != 0]
f_LPR, f_LPR_input, mask = encode_binary(f_LPR, p)
#f_LPR,mask = encode_con(f_LPR, 0.01)
f_LPR_h = read_header(path + "/data_encoded/phenotypes_age/father_LPR_headers_con.txt", mask)

m_LPR = read_con(path + "/data_encoded/input/mother_LPR_con.npy")
m_LPR = m_LPR[labels_names != '',:]
m_LPR = m_LPR[np.sum(geno_input, axis=1) != 0]
m_LPR, m_LPR_input, mask = encode_binary(m_LPR, p)
#m_LPR, mask = encode_con(m_LPR, 0.01)
m_LPR_h = read_header(path + "/data_encoded/phenotypes_age/mother_LPR_headers_con.txt", mask)

# combine parents and sibling
family_LPR = np.concatenate((f_LPR, m_LPR, sibling_pheno), axis=1)
family_LPR_h = np.concatenate((f_LPR_h, m_LPR_h, sibling_pheno_h))
family_LPR_input = np.concatenate((f_LPR_input, m_LPR_input, sibling_pheno_input), axis=1)

## Set paracombinemeters
analysis_type = "all"
version = 'v3'

sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')

first_age = np.load(path + "/data_encoded/included_patient_first_age.npy")
first_age = first_age[labels_names != '']
first_age = first_age[np.sum(geno_input, axis=1) != 0]

ages = np.load('/faststorage/jail/project/gentofte_projects/prediction/data_encoded/included_patient_age.npy')
ages = ages[labels_names != '']
ages = ages[np.sum(geno_input, axis=1) != 0]

labels_names = np.load(path + "/data_encoded/included_patient_labels.npy")
labels_names = np.where(labels_names=='BPD', 'BD', labels_names) 
#labels_names = np.where(labels_names == 'Anorexia','Back_pop', labels_names)
labels_names = labels_names[labels_names != '']
labels_names = labels_names[np.sum(geno_input, axis=1) != 0]
y = label_binarize(labels_names, classes=np.unique(labels_names))
labels = np.argmax(y, 1)

n_classes = y.shape[1]

labels_name_not_sort = list(np.unique(labels_names))
labels_name = ['Back_pop', 'ADHD', 'ASD', 'MDD', 'BD', 'SCZ']
# Prepare data
geno_input = geno_input[np.sum(geno_input, axis=1) != 0]
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
#cat_all = np.concatenate((base_cat_encoded_2, base_cat_encoded_7, base_cat_encoded_icd10), axis=1)

con_list = [F_pheno, severity_pheno, mbr, LPR, PRS]
n_con_shapes, mask, con_all = concat_con_list(con_list, mask)



all_raw = np.concatenate((con_all_raw, family_LPR_input, mbr_geno_input), axis=1)
data_df_raw = pd.DataFrame(all_raw, columns = np.concatenate((con_names,cat_names, emb_names)))

filtered = []
tmp_data = data_df_raw[np.concatenate((F_pheno_h, LPR_h))]
#new_age = []
new_age = np.load('/faststorage/jail/project/gentofte_projects/prediction/data_encoded/included_patient_age_corr_all.npy')
for j in range(len(first_age)):
    ep = tmp_data.loc[j,:]
    #p = np.array([math.ceil(elem) for elem in p])
    #if labels_names[j] == 'SCZ':
    #    print(first_age[j])
    #    print(list(p))
    ep[ep >= new_age[j]] = 0
    filtered.append(np.array(ep))
    
    #if first_age[j] == 100:
    #  new_age.append(ages[j])
    #else:
    #  new_age.append(first_age[j])

# Remove MDD
new_age_enc, mask = encode_con(np.array(new_age)[:, None], p)
filtered = np.array(filtered)

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
#cat_all = np.concatenate((base_cat_encoded_2, base_cat_encoded_7, base_cat_encoded_icd10), axis=1)

con_names = np.concatenate((filtered_h, ['age'], PRS_h, mbr_h))
con_list = [filtered_enc, new_age_enc, PRS, mbr]
n_con_shapes, mask, con_all = concat_con_list(con_list, mask)

#del sys.modules["multi_class"]
#from multi_class_emb import train_model, best_hyp_test, get_w_sampler, multi_acc, evaluate_model_nn, train_e, make_conf_matrix, test_model

RSEED = 42
skf_test = StratifiedKFold(n_splits=3, random_state = RSEED, shuffle=True)
#skf_val = StratifiedKFold(n_splits=3, random_state = RSEED)

EPOCHS = 15
NUM_FEATURES = [len(cat_names), len(con_names)]
cat_dims = [int(np.max(data_df.loc[:,col])) + 1 for col in emb_names]

#emb_dims = [(x, min(50, x // 2 + 1)) for x in cat_dims]
emb_dims = [(x, min(50,round((1.6 * x ** 0.56)))) for x in cat_dims]

NUM_CLASSES = len(np.unique(labels))
cuda = False
device = torch.device("cuda" if cuda == True else "cpu")
#BATCH_SIZE = 20
#LEARNING_RATE = 0.001
val_loss_min = 0
#val_loss_min = 100000
#batch_sizes = [64, 128]
batch_sizes = [128]
learning_rates = [0.0001]
#[128, 64],[512, 256]
#num_hiddens = [[128, 64], [256, 128]]
num_hiddens = [[128, 64]]

#batch_sizes = [50]
#learning_rates = [0.001]
#num_hiddens = [[256, 128]]
performance = []
best_hyps = []
test_predictions = []
test_probs = []
test_idx = []
models = []
nr = 1

batch_up = [5, 10]
learning_rates = [0.0001]

path = '/faststorage/jail/project/gentofte_projects/'
data_df = data_df_filtered

#all_data_names = pd.DataFrame(data_filtered_h)
#all_data_names.to_csv(path + "/prediction/included_names_all.txt", sep = "\n")

performance = np.load(path + "/prediction/" + analysis_type + "/nn_performance_" + version + "_" + analysis_type + ".npy",  allow_pickle=True)
test_predictions = np.load(path + "/prediction/" + analysis_type + "/nn_predictions_filtered_" + version + "_" + analysis_type + ".npy",  allow_pickle=True)
test_probs = np.load(path + "/prediction/" + analysis_type + "/nn_probs_filtered_" + version + "_" + analysis_type + ".npy",  allow_pickle=True)

models = []
for j in range(3):
    filename = path + "/prediction/" + analysis_type + "/nn_model_filtered_" + str(j) + "_" + version + "_" + analysis_type + ".pt"
    model = torch.load(filename)
    model.eval()
    models.append(model)

test_idx = []
for train_index_1, test_index in skf_test.split(data_df, labels):
   test_idx.append(test_index)

# Make collected performancece
colors = cycle(['lightskyblue','royalblue', 'darkblue', 'salmon', 'red', 'crimson', 'maroon'])
predictions_all = np.concatenate(test_predictions)
probs_all = np.concatenate(test_probs)
test_labels_all = np.concatenate([labels[test_idx[0]], labels[test_idx[1]],labels[test_idx[2]]])
test_eval = evaluate_model_nn(predictions_all, probs_all, test_labels_all, labels_name)

other_h = [i for i in filtered_h if i in LPR_h]
mental_h = [i for i in filtered_h if i in F_pheno_h]
all_geno_h = np.concatenate((geno_h,hla_pheno_h,PRS_h))

cat_names = np.concatenate((family_LPR_h,mbr_h))
emb_names = np.concatenate((MBR_pheno_h, hla_pheno_h, geno_h))
con_names = np.concatenate((filtered_h, ['age'], PRS_h))

data_names = [mental_h, other_h, ['age'], MBR_pheno_h, mbr_h, PRS_h, family_LPR_h, geno_h, hla_pheno_h, all_geno_h]
title_data = ['Psychiatric disorders', 'Other medical conditions', 'Age', 'MBR (categorical)', 'MBR (continuous)', 'PRS','Family diagnoses', 'Genomics', 'HLA data', 'All genetics']

colors_u = ['#90021C', '#EC1C1C','#E06161', '#FF9B9B', '#FF9B9B', '#84C3F7', '#4387BF', '#2669A1', '#022F90', '##022F99']
#colors_u = ['tomato', 'coral', 'lightblue', 'azure', 'navy']
bar_colors_all = dict()
for i,dn in enumerate(data_names):
    bar_colors_all[title_data[i]] = colors_u[i]

##mean_v = tmp_d[test_labels == 0].mean(0)
#if len(np.intersect1d(dn, PRS_h)) != 0 or len(np.intersect1d(dn, mbr_h)) != 0 or len(np.intersect1d(dn, other_h)) != 0  or len(np.intersect1d(dn, mental_h)) != 0:
combined_acc_df = defaultdict(list)
combined_mcc_df = defaultdict(list)
combined_auc_df = defaultdict(list)
for dt in range(len(data_names)):
    data_df_new = data_df.drop(data_names[dt], axis=1, inplace=False)
    cat_names_new = cat_names
    emb_names_new = emb_names
    con_names_new = con_names
    if len(np.intersect1d(data_names[dt], con_names)) != 0:
        con_names_new = [x for x in con_names if x not in data_names[dt]]
    if len(np.intersect1d(data_names[dt], cat_names)) != 0:
        cat_names_new = [x for x in cat_names if x not in data_names[dt]]
    if len(np.intersect1d(data_names[dt], emb_names)) != 0:
        emb_names_new = [x for x in emb_names if x not in data_names[dt]]
    NUM_FEATURES = [len(cat_names_new), len(con_names_new)]
    cat_dims = [int(np.max(data_df.loc[:,col])) + 1 for col in emb_names_new]
    emb_dims = [(x, min(50,round((1.6 * x ** 0.56)))) for x in cat_dims]
    nr = 0
    for train_index_1, test_index in skf_test.split(data_df_new, labels):
        acc = performance[nr][-1]
        mcc = performance[nr][-2]
        auc = performance[nr][-3]['roc']
        train_1, test = data_df_new.loc[train_index_1, :], data_df_new.loc[test_index, :]
        train_labels_1, test_labels = labels[train_index_1], labels[test_index]
        emb_test, cat_test, con_test = test.loc[:,emb_names_new], test.loc[:,cat_names_new], test.loc[:,con_names_new]
        test_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_test)).long(), torch.from_numpy(np.array(cat_test)).float(), torch.from_numpy(np.array(con_test)).float(),torch.from_numpy(test_labels).long())
        best_hyps_val = []
        epochs = []
        train, val, train_labels, val_labels = train_test_split(train_1, train_labels_1, stratify = train_labels_1, test_size = 0.1, random_state = RSEED)
        emb_train, cat_train, con_train= train.loc[:,emb_names_new], train.loc[:,cat_names_new], train.loc[:,con_names_new]
        train_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_train)).long(), torch.from_numpy(np.array(cat_train)).float(), torch.from_numpy(np.array(con_train)).float(),torch.from_numpy(train_labels).long())
        emb_val, cat_val, con_val = val.loc[:, emb_names_new], val.loc[:,cat_names_new], val.loc[:,con_names_new]
        val_dataset = ClassifierDataset(torch.from_numpy(np.array(emb_val)).long(), torch.from_numpy(np.array(cat_val)).float(), torch.from_numpy(np.array(con_val)).float(),torch.from_numpy(val_labels).long())
        weighted_sampler, class_weights = get_w_sampler(train_dataset, labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        val_loss_min = 0
        old_min = 0
        best_hyps_tmp = [batch_sizes[0], learning_rates[0], num_hiddens[0]]
        epoch = np.copy(EPOCHS)
        for BATCH_SIZE in batch_sizes:
            for LEARNING_RATE in learning_rates:
                for num_hidden in num_hiddens:
                    num_hidden1 = num_hidden[0]
                    num_hidden2 = num_hidden[1]
                    
                    val_loss_min, best_hyps_tmp, epoch, model = train_e(train_dataset, BATCH_SIZE, weighted_sampler, class_weights, NUM_FEATURES, NUM_CLASSES, num_hidden1, num_hidden2, EPOCHS, LEARNING_RATE, val_dataset, val_loss_min, best_hyps_tmp, epoch, labels, device, emb_dims, batch_up, loss_min=False)                    
                    if val_loss_min > old_min:
                        best_model = copy.deepcopy(model)
                        old_min = val_loss_min
                epochs.append(epoch)
        
        best_epochs = max(set(epochs), key = epochs.count)
        test_acc, test_loss, predictions, probs = test_model_new(best_model, test_dataset, device, labels)
        test_eval = evaluate_model_nn(np.array(predictions), np.array(probs), test_labels, labels_name)
        mcc_all = matthews_corrcoef(test_labels, np.array(predictions))
        combined_acc_df[title_data[dt]].append(np.abs(acc-test_acc))
        combined_mcc_df[title_data[dt]].append(np.abs(mcc-mcc_all))
        combined_auc_df[title_data[dt]].append(np.abs(auc-test_eval['roc']))
        filename = path + "/prediction/" + analysis_type + "/feature_new_train/nn_model_filtered_" + str(j) + "_" + version + "_" + analysis_type +  "_" + title_data[dt].replace(" ", "_") + ".pt"
        torch.save(best_model, filename)
        nr += 1


np.save(path + "/prediction/" + analysis_type + "/feature_new_train/nn_feature_importance_collected_" + version  + "_" + analysis_type +  "_acc.npy",combined_acc_df)
np.save(path + "/prediction/" + analysis_type + "/feature_new_train/nn_feature_importance_collected_" + version  + "_" + analysis_type +  "_mcc.npy",combined_mcc_df)
np.save(path + "/prediction/" + analysis_type + "/feature_new_train/nn_feature_importance_collected_" + version  + "_" + analysis_type +  "_auc.npy",combined_auc_df)

# path = 
# analysis_type = 'all'
# version = 'v3'
# combined_acc_df = np.load(path + "/prediction/" + analysis_type + "/feature_new_train/nn_feature_importance_collected_" + version  + "_" + analysis_type +  "_acc.npy", allow_pickle=True).item()
# df_ne = pd.DataFrame([combined_acc_df.keys(),[np.mean(v) for k,v in combined_acc_df.items()]]).T
# df_ne.columns = ['feature', 'importance']
# 
# df_ne.loc[:,'importance'] = (df_ne.loc[:,'importance'] / performance[1][-1]) * 100
# df_ne = df_ne.loc[:,['feature', 'importance']] .sort_values(by='importance', ascending = False)
