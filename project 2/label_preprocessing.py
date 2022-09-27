import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

file_path = '/home/HardDisk1/home/bispl/work/healthcare/physionet/files/mimiciii/1.4/'

curr_dir = os.getcwd()
noteevents_chunks = pd.read_csv(os.path.join(curr_dir,'filtered_noteevents.csv.gz'),chunksize=10000)
HADM_ID_list = []
for chunk in tqdm(noteevents_chunks):
    HADM_ID_list += list(chunk.HADM_ID.unique())

DIAGNOSES_ICD = pd.read_csv(file_path+'DIAGNOSES_ICD.csv.gz')
D_ICD_DIAGNOSES = pd.read_csv(file_path+'D_ICD_DIAGNOSES.csv.gz')
PROCEDURES_ICD = pd.read_csv(file_path+'PROCEDURES_ICD.csv.gz')
D_ICD_PROCEDURES = pd.read_csv(file_path+'D_ICD_PROCEDURES.csv.gz')
to_string_list = [DIAGNOSES_ICD,D_ICD_DIAGNOSES,PROCEDURES_ICD,D_ICD_PROCEDURES]

# ICD9 code to string
for df in to_string_list:
    df.ICD9_CODE = df.ICD9_CODE.apply(lambda x : str(x))

# Add Prefix
DIAGNOSES_ICD.ICD9_CODE = DIAGNOSES_ICD.ICD9_CODE.apply(lambda x : 'd' + x)
D_ICD_DIAGNOSES.ICD9_CODE = D_ICD_DIAGNOSES.ICD9_CODE.apply(lambda x : 'd' + x)
PROCEDURES_ICD.ICD9_CODE = PROCEDURES_ICD.ICD9_CODE.apply(lambda x : 'p' + x)
D_ICD_PROCEDURES.ICD9_CODE = D_ICD_PROCEDURES.ICD9_CODE.apply(lambda x : 'p' + x)

# ICD id to integer number
D_ICD = pd.concat([D_ICD_DIAGNOSES,D_ICD_PROCEDURES])
D_ICD_unique = list(map(lambda x : str(x),D_ICD.ICD9_CODE.unique()))
D_ICD_unique.sort()
D_ICD_id = dict(zip(D_ICD_unique,list(range(len(D_ICD_unique))))) # ID to integer dictionary ICD9 CODES

# ID to ICD9 Codes of string
ID_ICD9_CODE = pd.concat([DIAGNOSES_ICD[['HADM_ID','ICD9_CODE']],PROCEDURES_ICD[['HADM_ID','ICD9_CODE']]])
ID_ICD9_CODE = ID_ICD9_CODE[ID_ICD9_CODE.HADM_ID.isin(HADM_ID_list)]
ID_ICD9_CODE = ID_ICD9_CODE[ID_ICD9_CODE.ICD9_CODE.isin(D_ICD_unique)]
ID_ICD9_CODE = ID_ICD9_CODE.sort_values(by=['HADM_ID'], axis=0)
ID_ICD9_CODE.ICD9_CODE = ID_ICD9_CODE.ICD9_CODE.apply(lambda x : str(D_ICD_id[x]))
ID_ICD9_CODE = ID_ICD9_CODE.groupby('HADM_ID',as_index=False)['ICD9_CODE'].apply(lambda x: "%s" % ','.join(x)) # ID to Codes in Dataframe

# Label Preprocessing
code_num = len(D_ICD_unique)
def labels_to_one_hot(labels,code_num=code_num):
    one_hot_np = np.zeros(code_num,dtype=np.int8)
    labels = list(map(lambda x: int(x),labels.split(',')))
    one_hot_np[labels] = 1
    one_hot_np = list(map(str,one_hot_np))
    one_hot = ','.join(one_hot_np)
    return one_hot

y_train = open('y_train.txt','w')
y_test = open('y_test.txt','w')

for index, row in tqdm(ID_ICD9_CODE.iterrows()):
    hadm_id = row.HADM_ID
    icd9_code = labels_to_one_hot(row.ICD9_CODE)
    if (hadm_id % 10 == 8) or (hadm_id % 10 == 9) :
        y_test.write(str(hadm_id)+','+icd9_code+'\n')
    else :
        y_train.write(str(hadm_id)+','+icd9_code+'\n')
    