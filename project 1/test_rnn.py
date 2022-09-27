import os
import re
import gzip
import csv
import json
import time
import datetime
import math
import collections
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
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


num_item = len(ITEMID_list) + 1
num_ethn = len(ETHNICITY_list) + 1
num_adms = len(ADMISSION_TYPE_list) + 1
num_adlc = len(ADMISSION_LOCATION_list) + 1

train_X_dict = np.load('X_train_rnn.npy',allow_pickle=True)[()]
test_X_dict = np.load('X_test_rnn.npy',allow_pickle=True)[()]
train_y = np.load('y_train.npy')
test_y = np.load('y_test.npy')
del_key = []

sample_key = list(train_X_dict.keys())[0]
feature = train_X_dict[sample_key].keys()
feature = ['ADMISSION_TYPE','ADMISSION_LOCATION', 'ETHNICITY',  'ITEMID', 'VALUENUM']
item_value_dict = collections.defaultdict(list)
for ID in train_X_dict :
    patient = train_X_dict[ID]
    event = patient['Event']
    for iteration in event :
        data = event[iteration]
        item_value_dict[data['ITEMID']].append(float(data['VALUENUM']))
item_value_max = {}
item_value_min = {}

for item in item_value_dict:
    item_value_max[item] = np.max(item_value_dict[item])
    item_value_min[item] = np.min(item_value_dict[item])

def timedelta2float(t1,t2):
    if t1 >= t2 :
        return (t1-t2).seconds
    else :
        return 0
class RNN_Dataset(Dataset):
    def __init__(self,dictionary,target):
        self.dictionary = dictionary
        self.target = target
        self.IDs  = list(dictionary.keys())
        self.len = len(self.IDs)
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        ID = self.IDs[idx]
        patient = self.dictionary[ID]
        ctime_events = patient['Event']
        len_charttime = len(ctime_events)
        admns = patient['ADMISSION_TYPE']
        ethns = patient['ETHNICITY']
        adlcs = patient['ADMISSION_LOCATION']
        intime = patient['INTIME']
        admtime = patient['ADMITTIME']
        timedel = np.log(timedelta2float(intime,admtime)+1)
        features_list = []
        x_np = np.zeros((100,7)) + np.array([0,num_adms-1,num_ethn-1,num_adlc-1,num_item-1,0,0])
        for i,time in enumerate(ctime_events.keys()):            
            ctime = np.log(ctime_events[time]['CHARTTIME']+1)
            itemid = ctime_events[time]['ITEMID']
            valuenum = ctime_events[time]['VALUENUM']
            try : 
                valuenum = (valuenum-item_value_min[itemid])/(item_value_max[itemid]-item_value_min[itemid]+0.0001)
            except :
                valuenum = 0
            feature_np = np.array([ctime,admns,ethns,adlcs,itemid,valuenum,timedel])
            x_np[99-i] = feature_np

        # index 0 : ctime, 1:Admission type, 2:Ethnicity, 3:Admission Loc, 4 :Itemid 5:valuenum 6: In-Admit
        X = torch.from_numpy(x_np)
        y = torch.tensor(self.target[idx])

        return X,y


class GRU(nn.Module):
    def __init__(self,num_adms,num_ethn,num_item,num_adlc) :
        super(GRU,self).__init__()
        admns_embedding_size = 5
        ethns_embedding_size = 5
        adlcs_embedding_size = 5
        items_embedding_size = 30
        self.admns = nn.Embedding(num_adms,admns_embedding_size)
        self.ethns = nn.Embedding(num_ethn,ethns_embedding_size)
        self.adlcs = nn.Embedding(num_adlc,adlcs_embedding_size)
        self.items = nn.Embedding(num_item,items_embedding_size)
        input_size = adlcs_embedding_size + admns_embedding_size+ethns_embedding_size+items_embedding_size\
             + 1 + 1 + 1

        self.gru = nn.GRU(input_size,64,3,batch_first=True,dropout=0.2)
        self.out_layer = nn.Sequential(
            nn.Linear(64,1)
        )
    def forward(self,x):
        x = x.float()

        admns = self.admns(x[:,:,1].long())
        ethns = self.ethns(x[:,:,2].long())
        adlcs = self.adlcs(x[:,:,3].long())
        items = self.items(x[:,:,4].long())
        out = torch.cat([admns,ethns,adlcs,items,x[:,:,[0,5,6]]],dim=2)
        if torch.sum(x) == 0:
            out = out * 0 -1
        out,h = self.gru(out)
        out = out[:,-1]
        out = self.out_layer(out)

        return out
train_dataset = RNN_Dataset(train_X_dict,train_y)
train_loader = DataLoader(train_dataset,batch_size = 1,shuffle=True)
test_dataset = RNN_Dataset(test_X_dict,test_y)
test_loader = DataLoader(test_dataset,batch_size = 1,shuffle=False)
device = "cuda" if torch.cuda.is_available() else 'cpu'

gru = GRU(num_adms,num_ethn,num_item,num_adlc)
gru = gru.to(device)

checkpoint = torch.load('./saved_gru.pth')
gru.load_state_dict(checkpoint['model_state_dict'])

sys.stdout = open('20213205_rnn.txt','w')
print('20213205')
prediction = []
target = []
for i,(X,y) in enumerate(train_loader):
    X = X.to(device)
    y = y.to(device)
    pred = gru(X)
    prediction.append(torch.sigmoid(pred))
    target.append(y)
prediction = torch.cat(prediction,dim=0)
target = torch.cat(target,dim=0)
AUROC = roc_auc_score(target.cpu().detach().numpy(),prediction.cpu().detach().numpy())
AUPRC = average_precision_score(target.cpu().detach().numpy(),prediction.cpu().detach().numpy())
print(AUROC)
print(AUPRC)
prediction = []
target = []
for i,(X,y) in enumerate(test_loader):
    X = X.to(device)
    y = y.to(device)
    pred = gru(X)
    prediction.append(torch.sigmoid(pred))
    target.append(y)
prediction = torch.cat(prediction,dim=0)
target = torch.cat(target,dim=0)
AUROC = roc_auc_score(target.cpu().detach().numpy(),prediction.cpu().detach().numpy())
AUPRC = average_precision_score(target.cpu().detach().numpy(),prediction.cpu().detach().numpy())
print(AUROC)
print(AUPRC)