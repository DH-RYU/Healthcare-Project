import os
import re
import gzip
import csv
import json
import time
import datetime
import math
import pandas as pd
import pickle
import numpy as np
import sys
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler


admission_df = pd.read_csv('ADMISSIONS.csv.gz')
d_items_df = pd.read_csv('D_ITEMS.csv.gz')

ITEMID_list = list(d_items_df.ITEMID.unique())
ITEMID_dict = dict(zip(ITEMID_list,list(range(len(ITEMID_list)))))
ETHNICITY_list = list(admission_df.ETHNICITY.unique())
ETHNICITY_dict = dict(zip(ETHNICITY_list,list(range(len(ETHNICITY_list)))))
ADMISSION_TYPE_list = list(admission_df.ADMISSION_TYPE.unique())
ADMISSION_TYPE_dict = dict(zip(ADMISSION_TYPE_list,list(range(len(ADMISSION_TYPE_list)))))
ADMISSION_LOCATION_list = list(admission_df.ADMISSION_LOCATION.unique())
ADMISSION_LOCATION_dict = dict(zip(ADMISSION_LOCATION_list,list(range(len(ADMISSION_LOCATION_list)))))

num_item = len(ITEMID_list)
num_ethn = len(ETHNICITY_list)
num_adms = len(ADMISSION_TYPE_list)
num_adlc = len(ADMISSION_LOCATION_list)

train_X_dict = np.load('X_train_logistic.npy',allow_pickle=True)[()]
test_X_dict = np.load('X_test_logistic.npy',allow_pickle=True)[()]
train_y = np.load('y_train.npy')
test_y = np.load('y_test.npy')

def timedelta2float(t1,t2):
    if t1 >= t2 :
        return (t1-t2).seconds
    else :
        return 0
def dict_to_matrix(dictionary):
    N = len(dictionary)
    M = 1 + num_item + num_ethn + num_adms + num_adlc # 1 for admittime - intime
    matrix = np.zeros((N,M))
    scaler = MinMaxScaler()
    for i,ID in enumerate(dictionary.keys()) :
        items = np.zeros(num_item)
        admns = np.zeros(num_adms)
        ethns = np.zeros(num_ethn)
        adlcs = np.zeros(num_adlc)
        timedel = np.zeros(1)

        items[dictionary[ID]['ITEMID']] += dictionary[ID]['VALUENUM']
        admns[dictionary[ID]['ADMISSION_TYPE']] += 1
        ethns[dictionary[ID]['ETHNICITY']] += 1
        adlcs[dictionary[ID]['ADMISSION_LOCATION']] += 1
        timedel[0] = np.log(timedelta2float(dictionary[ID]['INTIME'],dictionary[ID]['ADMITTIME'])+1)
        ID_feature = np.concatenate([items,timedel,admns,ethns,adlcs],axis=0)
        matrix[i] = ID_feature
    matrix[:,:num_item+1] = scaler.fit_transform(matrix[:,:num_item+1])
    return matrix

model = pickle.load(open('./logistic.pkl','rb'))
train_X = dict_to_matrix(train_X_dict)
test_X = dict_to_matrix(test_X_dict)
pred = model.predict_proba(train_X)[:,1]
sys.stdout = open('20213205_logistic_regression.txt','w')

print('20213205')
print(roc_auc_score(train_y,pred))
print(average_precision_score(train_y,pred))
pred = model.predict_proba(test_X)[:,1]
print(roc_auc_score(test_y,pred))
print(average_precision_score(test_y,pred))