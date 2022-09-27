import numpy as np
import pandas as pd
import pickle as pkl
import random
import collections
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
from einops import rearrange
from math import floor
class Text_Dataset(Dataset):
    def __init__(self,X_list,y_list,num_words,sentence_len=1000):
        self.X_list = X_list
        self.y_list = y_list
        self.num_words = num_words
        self.sentence_len = sentence_len

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self,idx):
        X = self.X_list[idx]
        y = self.y_list[idx]
        if len(X)>self.sentence_len:
            r = random.randint(0,len(X)-self.sentence_len)
            X = X[r:r+self.sentence_len]
        else :
            r = self.sentence_len - len(X)
            ones = np.ones(r,dtype=int)
            X = np.concatenate([ones*(self.num_words+1),X]) # zero padding
        
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X,y

class CAML(nn.Module):
    def __init__(self,num_words,sentence_len = 2500 ,output_dim = 18000,embedding_dim=100,hidden_channels=100,kernel_size=3):
        super(CAML,self).__init__()
        self.num_words = num_words 
        self.sentence_len = sentence_len
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(0.2)

        self.word_embedding = nn.Embedding(self.num_words+2,self.embedding_dim,padding_idx=0) # two for unknown words and zero padding
        self.conv = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_channels, kernel_size=self.kernel_size,padding=int(floor(kernel_size/2))) # batch x embedding_size x words => batch x hidden_channel x words-2
        self.u = nn.Linear(self.hidden_channels,self.output_dim)
        self.out_layer = nn.Linear(self.hidden_channels,self.output_dim) # No padding
        xavier_uniform_(self.conv.weight)
        xavier_uniform_(self.u.weight)
        xavier_uniform_(self.out_layer.weight)

    def forward(self,x):
        x = self.word_embedding(x)
        x = self.dropout(x)
        x = x.transpose(1,2) # batch x embedding x word
        
        x = torch.tanh(self.conv(x).transpose(1,2))

        alpha = F.softmax(self.u.weight.matmul(x.transpose(1,2)), dim=2)
        m = alpha.matmul(x)
        y = self.out_layer.weight.mul(m).sum(dim=2).add(self.out_layer.bias)
        return y # return batch x output_size
X_train = open('./X_train_tmp.txt','r')
X_test = open('./X_test_tmp.txt','r')
y_train = open('./y_train.txt','r')
y_test = open('./y_test.txt','r')

def preprocess(data):
    data_list = []
    for line in tqdm(data.readlines()):
        line = line.strip()
        line = line.split(',')[1:]
        line = np.array(list(map(int,line)))
        data_list.append(line)
    return data_list
X_train = preprocess(X_train)
y_train = preprocess(y_train)
# X_test = preprocess(X_test)
# y_test = preprocess(y_test)

with open('./word_id_dict_0.pkl', 'rb') as fr:
    word_id_dict = pkl.load(fr)
output_dim = len(y_train[0])
num_words = len(word_id_dict)
sentence_len = 2000
embedding_dim = 100
hidden_channels = 50
kernel_size = 9
batch_size = 2

train_dataset = Text_Dataset(X_train,y_train,num_words=num_words,sentence_len=sentence_len)
train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
test_dataset = Text_Dataset(X_test,y_test,num_words=num_words,sentence_len=sentence_len)
test_loader = DataLoader(test_dataset,batch_size = 1, shuffle=False)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = CAML(num_words=num_words,sentence_len=sentence_len,output_dim=output_dim,embedding_dim=embedding_dim,hidden_channels=hidden_channels,kernel_size=kernel_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-04)
os.makedirs("./saved_model",exist_ok=True)

model.train()
for epoch in range(300):
    train_loss = 0

    model.train()
    prediction_list = []
    labels_list = []
    for i,(X,y) in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        prediction = model.forward(X)
        loss = criterion(prediction,y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if len(prediction_list) < 200 :
            prediction = torch.sigmoid(prediction)
            prediction_list.append(prediction.cpu().detach().numpy())
            labels_list.append(y.cpu().detach().numpy())

        if i % 1000 == 199 :
            labels_np = np.concatenate(labels_list,axis=0)
            prediction_np = np.concatenate(prediction_list,axis=0)
            print("\n")
            print(np.sum(np.round(prediction_np)))
            print(np.sum(labels_np))
            print(train_loss/200)
            train_loss = 0
            idx = np.argwhere(np.all(labels_np == 0, axis=0))
            labels_np = np.delete(labels_np, idx, axis=1)
            prediction_np = np.delete(prediction_np, idx, axis=1)

            AUROC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
            AUROC_micro = roc_auc_score(labels_np,prediction_np,average='micro')
            AUPRC_macro = average_precision_score(labels_np,prediction_np,average='macro')
            AUPRC_micro = average_precision_score(labels_np,prediction_np,average='micro')

            print(AUROC_macro)
            print(AUROC_micro)
            print(AUPRC_macro)
            print(AUPRC_micro)

            prediction_list = []
            labels_list = []
    prediction_list = []
    labels_list = []
    model.eval()
    torch.save({
        'model_state_dict' : model.state_dict()
    },'./saved_model/model_'+str(epoch)+'.pth')
    # for i,(X,y) in tqdm(enumerate(test_loader)):
    #     X = X.to(device)
    #     y = y.to(device)
    #     prediction = model(X)
    #     prediction_list.append(prediction.cpu().detach().numpy())
    #     labels_list.append(y.cpu().detach().numpy())
    
    # labels_np = np.concatenate(labels_list,axis=0)
    # prediction_np = np.concatenate(prediction_list,axis=0)

    # idx = np.argwhere(np.all(labels_np == 0, axis=0))
    # labels_np = np.delete(labels_np, idx, axis=1)
    # prediction_np = np.delete(prediction_np, idx, axis=1)

    # AUROC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
    # AUROC_micro = roc_auc_score(labels_np,prediction_np,average='micro')
    # AUPRC_macro = average_precision_score(labels_np,prediction_np,average='macro')
    # AUPRC_micro = average_precision_score(labels_np,prediction_np,average='micro')

    # print(AUROC_macro)
    # print(AUROC_micro)
    # print(AUPRC_macro)
    # print(AUPRC_micro)

        

