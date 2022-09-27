import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

file_path = '/home/HardDisk1/home/bispl/work/healthcare/physionet/files/mimiciii/1.4/'

noteevents_chunks = pd.read_csv(file_path+'NOTEEVENTS.csv.gz',chunksize=10000)

chunk_list = []
keep_cols = ['HADM_ID','TEXT']
for i,chunk in tqdm(enumerate(noteevents_chunks)) :
    chunk = chunk[chunk.HADM_ID.notna()]
    chunk = chunk[chunk.CATEGORY == 'Discharge summary']
    chunk = chunk[chunk.DESCRIPTION == "Report"]
    chunk_list.append(chunk[keep_cols])
filtered_noteevents = pd.concat(chunk_list)
filtered_noteevents.to_csv('./filtered_noteevents.csv.gz',index=False)
