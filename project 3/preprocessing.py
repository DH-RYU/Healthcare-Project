import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as F_t

from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle as pkl
from PIL import Image

csv_path = './physionet.org/files/mimic-cxr-jpg/2.0.0' 
file_path = './physionet.org/files/mimic-cxr-jpg/2.0.0/files' 
metadata_path = os.path.join(csv_path,'mimic-cxr-2.0.0-metadata.csv.gz')
negbio_path = os.path.join(csv_path,'mimic-cxr-2.0.0-negbio.csv.gz')

# metadata.csv tells you that it's AP view or not
# dicom_id : actually file id,study_id, ViewPosition
# negbio.csv contains multilabels

metadata_df = pd.read_csv(metadata_path)

metadata_df = metadata_df[metadata_df.ViewPosition=='AP']
subject_id_list = list(metadata_df.subject_id.unique())
study_id_list = list(metadata_df.study_id.unique())
print(metadata_df.columns)
print(metadata_df.head(10))
negbio_df = pd.read_csv(negbio_path)
print(negbio_df.columns)
y_test = open('./y_test.txt','w')
y_train = open('./y_train.txt','w') 
X_test = open('./X_test.pkl','wb')
X_train = open('./X_train.pkl','wb')
test_list = []
train_list = []
count = 0
for sb_id in tqdm(subject_id_list) :
    count += 1
    sb_df = negbio_df.loc[negbio_df.subject_id==sb_id,:]
    sb_df = sb_df.loc[negbio_df.study_id.isin(study_id_list),:]
    sb_df = sb_df.sort_values(by=['study_id'])
    selected_study_id = sb_df.iloc[0,:]

    selected_study_id = selected_study_id.fillna(0)
    selected_study_id = selected_study_id.replace(-1,0)
    selected_study_id = selected_study_id.astype(int)
    selected_study_id = selected_study_id.astype(str)
    data = selected_study_id.tolist()[1:]

    st_id = selected_study_id.study_id
    st_id_meta = metadata_df[metadata_df.study_id.astype(int)==int(st_id)]
    st_id_meta = st_id_meta.sort_values(by=['dicom_id'])
    di_id = st_id_meta.iloc[0,0]
    img_path = os.path.join(file_path,'p'+str(sb_id)[:2],'p'+str(sb_id),'s'+str(st_id),di_id+'.jpg')
    img = Image.open(img_path)
    img = img.resize((256,256))
    img_np = np.array(img)
    if int(data[0])%10 == 8 or int(data[0])%10 == 9 :
        data_str = ','.join(data)
        y_test.write(data_str+'\n')
        test_list.append([img_np,data[0]])
    else :
        data_str = ','.join(data)
        y_train.write(data_str+'\n')
        train_list.append([img_np,data[0]])

pkl.dump(test_list,X_test)
pkl.dump(train_list,X_train)
X_test.close()
X_train.close()
y_test.close()
y_train.close()