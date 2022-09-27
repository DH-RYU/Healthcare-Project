import os
import gzip
import csv
import json
import time
import pandas as pd
from tqdm.auto import tqdm
from xgboost import XGBClassifier

admission = 'ADMISSIONS.csv.gz'
'''
ROW_ID	                INT
SUBJECT_ID	            INT
HADM_ID	                INT
ADMITTIME	            TIMESTAMP(0)
DISCHTIME	            TIMESTAMP(0)
DEATHTIME	            TIMESTAMP(0)
ADMISSION_TYPE	        VARCHAR(50)
ADMISSION_LOCATION	    VARCHAR(50)
DISCHARGE_LOCATION	    VARCHAR(50)
INSURANCE	            VARCHAR(255)
LANGUAGE	            VARCHAR(10)
RELIGION	            VARCHAR(50)
MARITAL_STATUS	        VARCHAR(50)
ETHNICITY	            VARCHAR(200)
EDREGTIME	            TIMESTAMP(0)
EDOUTTIME	            TIMESTAMP(0)
DIAGNOSIS	            VARCHAR(300)
HOSPITAL_EXPIRE_FLAG	TINYINT
HAS_CHARTEVENTS_DATA	TINYINT
'''

chartevent = 'CHARTEVENTS.csv.gz'
'''
ROW_ID	        INT
SUBJECT_ID	    NUMBER(7,0)
HADM_ID	        NUMBER(7,0)
ICUSTAY_ID	    NUMBER(7,0)
ITEMID	        NUMBER(7,0)
CHARTTIME	    DATE
STORETIME	    DATE
CGID	        NUMBER(7,0)
VALUE	        VARCHAR2(200 BYTE)
VALUENUM	    NUMBER
VALUEUOM	    VARCHAR2(20 BYTE)
WARNING	        NUMBER(1,0)
ERROR	        NUMBER(1,0)
RESULTSTATUS	VARCHAR2(20 BYTE)
STOPPED	        VARCHAR2(20 BYTE)
Detail
'''

icustays = 'ICUSTAYS.csv.gz'
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
curr_dir = os.getcwd()

icustays_chunk_container = pd.read_csv(icustays, chunksize=10000)

chunk_list = []
for i,chunk in enumerate(icustays_chunk_container) :
    chunk = chunk[chunk.LOS >= 1]
    chunk = chunk[chunk.LOS <= 2]
    chunk_list.append(chunk)
filtered_icustays = pd.concat(chunk_list)
icustay_id_list = list(filtered_icustays.ICUSTAY_ID.unique())

print(len(icustay_id_list))
chartevent_chunk_container = pd.read_csv(chartevent, chunksize=1000000)


chunk_list = []
for i,chunk in tqdm(enumerate(chartevent_chunk_container)) : 
    chunk = chunk[chunk.ICUSTAY_ID.isin(icustay_id_list)]
    chunk_list.append(chunk)
filtered_chartevent = pd.concat(chunk_list)
filtered_icustays.to_csv('./filtered_icustays.csv.gz',index=False)
filtered_chartevent.to_csv('./filtered_chartevents.csv.gz',index=False)