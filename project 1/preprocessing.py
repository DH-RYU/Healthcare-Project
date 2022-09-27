import os
import gzip
import csv
import json
import time
import datetime
import pandas as pd
from tqdm.auto import tqdm

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

chartevent = 'filtered_chartevents.csv.gz'
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

chartevent_df = pd.read_csv(chartevent)
chartevent_df.CHARTTIME = chartevent_df.CHARTTIME.apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
icustays_df = pd.read_csv(icustays)
icustays_df.INTIME = icustays_df.INTIME.apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

icu_ID_list = list(icustays_df.ICUSTAY_ID.unique()) # 
print(len(icu_ID_list))
threshold_time = datetime.timedelta(hours=3)
chunk_list = []
for ID in tqdm(icu_ID_list) :
    chartevent_ID = chartevent_df[chartevent_df.ICUSTAY_ID==ID]
    icustays_ID = icustays_df[icustays_df.ICUSTAY_ID == ID]
    time_diff = chartevent_ID.CHARTTIME - icustays_ID.INTIME.item()
    chartevent_ID = chartevent_ID.loc[time_diff < threshold_time,:]
    chunk_list.append(chartevent_ID)
filtered_df = pd.concat(chunk_list)
filtered_df.to_csv('./filtered_chartevents_final_1.csv.gz',index=False)
