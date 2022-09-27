import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
import torch.nn as nn
import torch
from nltk.tokenize import word_tokenize
# from sklearn.metrics import roc_auc_score,average_precision_score
# from sklearn.datasets import make_multilabel_classification
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# X, y = make_multilabel_classification(random_state=0)
# inner_clf = LogisticRegression(solver="liblinear", random_state=0)
# clf = MultiOutputClassifier(inner_clf).fit(X, y)
# y_score = np.transpose([y_pred[:, 1] for y_pred in clf.predict_proba(X)])



# labels_np = np.random.randint(0,2,(10,20))
# prediction_np = np.random.random((10,20))
# idx = np.argwhere(np.all(labels_np == 0, axis=0))
# print(labels_np)
# labels_np = np.delete(labels_np, idx, axis=1)
# prediction_np = np.delete(prediction_np, idx, axis=1)

# AUROC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
# AUROC_micro = average_precision_score(labels_np,prediction_np,average='micro')
# AUPRC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
# AUPRC_micro = average_precision_score(labels_np,prediction_np,average='micro')

# print(AUROC_macro)
# print(AUROC_micro)
# print(AUPRC_macro)
# print(AUPRC_micro)
# print('\n')
# a = [1,2,3]
# b = [1,2,4]
import re

# text = "a dfs d wr g dszh f he's food 20 20g 30 30g g30g"
# p = re.compile("[a-zA-Z0-9_']+")
# text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
# print(text)
# print(p.findall(text))
import pickle as pkl
# with open('./word_id_dict_0.pkl', 'rb') as fr:
#     word_id_dict = pkl.load(fr)
# print(len(word_id_dict))
# print(word_id_dict.keys())

# a = np.zeros(10,dtype=np.int)

# a[[2,1,4]] += 1
# print(a)
# a = list(map(str,a))
# print(a)
# a = ','.join(a)
# print(a)

y_train = open('X_train.txt','r')

y = y_train.readlines()
count = 0 
for l in y :
 
    count += 1
print(count)