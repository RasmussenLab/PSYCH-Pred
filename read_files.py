import os, sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_cat(file):
   data = np.load(file)
   data = data.astype(np.float32)
   data_input = data.reshape(data.shape[0], -1)
   data_label = np.argmax(data, 2)
   data_label = data_label.astype('float')
   data_label[data.sum(2) == 0] = -1
   data_label = data_label + 1
   
   return data, data_label

def read_con(file):
   data = np.load(file)
   data = data.astype(np.float32)
   
   return data

def read_header(file, mask=None):
   with open(file, "r") as f:
      h = list()
      for line in f:
         h.append(line.rstrip())
   
   if not mask is None:
      h = np.array(h)
      h = h[mask]
   
   return h

def remove_not_obs_cat(pheno, ind, h, p=0.01):
   pheno = pheno[:,~np.all(ind == ind[0,:], axis = 0)]
   h = np.array(h)[~np.all(ind == ind[0,:], axis = 0)]
   ind = ind[:,~np.all(ind == ind[0,:], axis = 0)]
   
   ind_tmp = np.copy(ind)
   ind_tmp[ind_tmp == 1] = 0
   pheno = pheno[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   h = h[np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   ind = ind[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   
   return pheno, ind, h

def encode_con(raw_input, p = 0.01, min_v = 0, max_v = 1):
   
   matrix = np.array(raw_input)
   tmp_matrix = np.copy(matrix)
   tmp_matrix[np.isnan(tmp_matrix)] = 0
   
   # remove less than p% observations
   mask_col = np.count_nonzero(tmp_matrix, 0) > (tmp_matrix.shape[0] * p)
   
   scaler = MinMaxScaler((min_v, max_v))
   data_input = scaler.fit_transform(matrix)
   
   # remove 0 variance
   std = np.nanstd(data_input, axis=0)
   mask_col &= std != 0
  
   mean = np.nanmean(data_input, axis = 0)
   data_input = np.where(np.isnan(matrix), mean, data_input)
   data_input = data_input[:,mask_col]
   
   return data_input, mask_col

def encode_binary(raw_input, p = 0.01):
   
   matrix = np.array(raw_input)
   tmp_matrix = np.copy(matrix)
   tmp_matrix[np.isnan(tmp_matrix)] = 0
   
   # remove less than p% observations
   mask_col = np.count_nonzero(tmp_matrix, 0) > (tmp_matrix.shape[0] * p)
   data_input = tmp_matrix
   
   # remove 0 variance
   std = np.nanstd(data_input, axis=0)
   mask_col &= std != 0
   data_input = data_input[:,mask_col]
   
   data_input[data_input != 0] = 1
   
   data_input[np.isnan(matrix[:,mask_col])] = np.nan
   
   data_input = encode_cat(data_input, num_classes = 2, uniques = [0,1], na = np.nan)
   data_label = np.argmax(data_input, 2)
   data_label = data_label.astype('float')
   data_label[data_input.sum(2) == 0] = -1
   data_label = data_label + 1
   
   return data_input, data_label, mask_col

def concat_cat_list(cat_list):
  n_cat = 0
  cat_shapes = list()
  first = 0
 
  for cat_d in cat_list:
    cat_shapes.append(cat_d.shape)
    cat_input = cat_d.reshape(cat_d.shape[0], -1)
   
    if first == 0:
      cat_all = cat_input
      del cat_input
      first = 1
    else:
      cat_all = np.concatenate((cat_all, cat_input), axis=1)
 
  # Make mask for patients with no measurements
  catsum = cat_all.sum(axis=1)
  mask = catsum > 1
  del catsum
  return cat_shapes, mask, cat_all

def concat_con_list(con_list, mask=[]):
  n_con_shapes = []
 
  first = 0
  for con_d in con_list:
    con_d = con_d.astype(np.float32)
    n_con_shapes.append(con_d.shape[1])
     
    if first == 0:
      con_all = con_d
      first = 1
    else:
      con_all = np.concatenate((con_all, con_d), axis=1)
 
  consum = con_all.sum(axis=1)
  if len(mask) == 0:
      mask = consum != 0
  else:
      mask &= consum != 0
  del consum
  return n_con_shapes, mask, con_all
