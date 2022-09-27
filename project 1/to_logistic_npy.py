import os
import re
import gzip
import csv
import json
import time
import datetime
import math
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder

admission = 'ADMISSIONS.csv.gz'
'''
ROW_ID	                INT
SUBJECT_ID	            INT
HADM_ID	                INT
ADMITTIME	            TIMESTAMP(0)    INTIME에서 빼서 사용
DISCHTIME	            TIMESTAMP(0)
DEATHTIME	            TIMESTAMP(0)
ADMISSION_TYPE	        VARCHAR(50)
ADMISSION_LOCATION	    VARCHAR(50)     9
DISCHARGE_LOCATION	    VARCHAR(50)
INSURANCE	            VARCHAR(255)
LANGUAGE	            VARCHAR(10)
RELIGION	            VARCHAR(50)
MARITAL_STATUS	        VARCHAR(50)
ETHNICITY	            VARCHAR(200)    use
EDREGTIME	            TIMESTAMP(0)    
EDOUTTIME	            TIMESTAMP(0)
DIAGNOSIS	            VARCHAR(300)
HOSPITAL_EXPIRE_FLAG	TINYINT
HAS_CHARTEVENTS_DATA	TINYINT
'''

chartevent = 'filtered_chartevents_final.csv.gz'
'''
ROW_ID	        INT
SUBJECT_ID	    NUMBER(7,0)
HADM_ID	        NUMBER(7,0)
ICUSTAY_ID	    NUMBER(7,0)
ITEMID	        NUMBER(7,0)     use
CHARTTIME	    DATE            use
STORETIME	    DATE
CGID	        NUMBER(7,0)
VALUE	        VARCHAR2(200 BYTE)
VALUENUM	    NUMBER          use
VALUEUOM	    VARCHAR2(20 BYTE)
WARNING	        NUMBER(1,0)
ERROR	        NUMBER(1,0)
RESULTSTATUS	VARCHAR2(20 BYTE)
STOPPED	        VARCHAR2(20 BYTE)
Detail
'''

icustays = 'filtered_icustays.csv.gz'
'''
ROW_ID	        INT
SUBJECT_ID	    INT
HADM_ID	        INT
ICUSTAY_ID	    INT
DBSOURCE	    VARCHAR(20)
FIRST_CAREUNIT	VARCHAR(20)
LAST_CAREUNIT	VARCHAR(20)
FIRST_WARDID	SMALLINT
LAST_WARDID	    SMALLINT
INTIME	        TIMESTAMP(0)
OUTTIME	        TIMESTAMP(0)
LOS	            DOUBLE
'''

patient = {}

admission_df = pd.read_csv(admission)
chartevent_df = pd.read_csv(chartevent)
icustays_df = pd.read_csv(icustays)
d_items_df = pd.read_csv('D_ITEMS.csv.gz')
icustays_df.INTIME = icustays_df.INTIME.apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
icustays_df.OUTTIME = icustays_df.OUTTIME.apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

ITEMID_list = list(d_items_df.ITEMID.unique())
ITEMID_dict = dict(zip(ITEMID_list,list(range(len(ITEMID_list)))))
ETHNICITY_list = list(admission_df.ETHNICITY.unique())
ETHNICITY_dict = dict(zip(ETHNICITY_list,list(range(len(ETHNICITY_list)))))
ADMISSION_TYPE_list = list(admission_df.ADMISSION_TYPE.unique())
ADMISSION_TYPE_dict = dict(zip(ADMISSION_TYPE_list,list(range(len(ADMISSION_TYPE_list)))))
ADMISSION_LOCATION_list = list(admission_df.ADMISSION_LOCATION.unique())
ADMISSION_LOCATION_dict = dict(zip(ADMISSION_LOCATION_list,list(range(len(ADMISSION_LOCATION_list)))))
icu_ID_list = list(icustays_df.ICUSTAY_ID.unique())
icu_ID_list.sort()
# ID ends with 8 or 9 should be test dataset

train_icu_ID_list = []
test_icu_ID_list = []

for ID in icu_ID_list:
    if ID % 10 == 9 or ID % 10 == 8 :
        test_icu_ID_list.append(ID)
    else :
        train_icu_ID_list.append(ID)
def timedelta2float(t1,t2):
    if t1 >= t2 :
        return (t1-t2).seconds
    else :
        return 0
def str_to_time(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
def make_files(ID_list,mode):
    patient = {}
    for ID in tqdm(ID_list) :
        ID = int(ID)
        icustays_data = icustays_df[icustays_df.ICUSTAY_ID==ID]
        patient[ID] = {}
        patient[ID]['HADM_ID'] = int(icustays_data.HADM_ID.item())
        # patient[ID]['FIRST_CAREUNIT'] = icustays_data.FIRST_CAREUNIT.item()
        # patient[ID]['LAST_CAREUNIT'] = icustays_data.LAST_CAREUNIT.item()
        # patient[ID]['FIRST_WARDID'] = icustays_data.FIRST_WARDID.item()
        # patient[ID]['LAST_WARDID'] = icustays_data.LAST_WARDID.item()
        patient[ID]['INTIME'] = icustays_data.INTIME.item()
        patient[ID]['OUTTIME'] = icustays_data.OUTTIME.item()

        admission_data = admission_df[admission_df.HADM_ID==patient[ID]['HADM_ID']]

        patient[ID]['DEATHTIME'] = admission_data.DEATHTIME.item() # DEATHTIME should be before OUTTIME
        patient[ID]['ADMITTIME'] = str_to_time(admission_data.ADMITTIME.item())

        patient[ID]['ADMISSION_TYPE'] = ADMISSION_TYPE_dict[admission_data.ADMISSION_TYPE.item()]
        patient[ID]['ADMISSION_LOCATION'] = ADMISSION_LOCATION_dict[admission_data.ADMISSION_LOCATION.item()]
        patient[ID]['ETHNICITY'] = ETHNICITY_dict[admission_data.ETHNICITY.item()]
        try : 
            if math.isnan(patient[ID]['DEATHTIME']) :
                patient[ID]['label'] = 0
        except : 
            patient[ID]['DEATHTIME'] = str_to_time(patient[ID]['DEATHTIME'])
            if (patient[ID]['DEATHTIME'] >= patient[ID]['INTIME']) and (patient[ID]['DEATHTIME'] <= patient[ID]['OUTTIME']) :
                patient[ID]['label'] = 1
            else :
                patient[ID]['label'] = 0
        chartevent_data = chartevent_df[chartevent_df.ICUSTAY_ID==ID]
        chartevent_data = chartevent_data.sort_values("CHARTTIME")
        # ITEMID and VALUENUM first
        ITEMID = chartevent_data.ITEMID.apply(lambda x : ITEMID_dict[x]).to_numpy()
        VALUENUM = chartevent_data.VALUENUM.to_numpy()
        VALUENUM = np.nan_to_num(VALUENUM)
        patient[ID]['ITEMID'] = []
        patient[ID]['VALUENUM'] = []
        for item in set(ITEMID) :
            patient[ID]['ITEMID'].append(item)
            item_idx = np.where(ITEMID == item)[0]
            value_avg = np.mean(VALUENUM[item_idx])
            patient[ID]['VALUENUM'].append(value_avg)

    np.save(mode + "_logistic.npy", patient)
    np.save(mode + "_y.npy",np.array([patient[k]['label'] for k in patient]))
make_files(train_icu_ID_list,'train')
make_files(test_icu_ID_list,'test')
# d1 = np.load("train_logistic.npy",allow_pickle=True)[()]
# d2 = np.load("train_y.npy",allow_pickle=True)