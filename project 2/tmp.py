import numpy as np
import pandas as pd
import pickle as pkl
import random
import collections
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
from einops import rearrange
from math import floor
from dataproc import extract_wvs
from gensim.models import KeyedVectors,Word2vec
from constants import *

class Text_Dataset(Dataset):
    def __init__(self,X_list,y_list,num_words,sentence_len=1000):
        self.X_list = X_list
        self.y_list = y_list
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
            X = np.concatenate([X,ones*(self.num_words+1)])
        
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X,y

class CAML(nn.Module):
    def __init__(self,num_words,embed_file,sentence_len = 1000 ,output_dim = 18000,embedding_dim=100,hidden_channels=10,kernel_size=3):
        super(CAML,self).__init__()
        self.num_words = num_words 
        self.sentence_len = sentence_len
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # Embedding
        print("loading pretrained embeddings...")
        W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
        self.word_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.word_embedding.weight.data = W.clone()

        # Layers
        self.conv = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_channels, kernel_size=self.kernel_size,padding=int(floor(self.kernel_size/2)))
        self.u = nn.Linear(self.hidden_channels,self.output_dim)
        self.out_layer = nn.Linear(self.hidden_channels,self.output_dim) # No padding
        xavier_uniform(self.conv.weight)
        xavier_uniform(self.u.weight)
        xavier_uniform(self.out_layer.weight)

    def forward(self,x):
        emb = self.word_embedding(x)
        emb = rearrange(emb,"b w e -> b e w") # batch x embedding x word
        H = self.conv(emb)
        H = F.tanh(H)
        H = rearrange(H,"b c w -> b w c") # batch x word x hidden_channels
        attns = F.softmax(self.u.weight.matmul(H.transpose(1,2)), dim=2)
        m = attns.matmul(H)
        out = self.out_layer.weight.mul(m).sum(dim=2).add(self.out_layer.bias)
        out = torch.sigmoid(out).squeeze(-1)
        return out 
embed_file = "./"
extract_wvs.load_embeddings(embed_file)
caml = CAML(num_words=10)
a = torch.randint(0,10,(5,100))
print(caml(a).shape)
# X_train = open('./X_train.txt','r')
# X_test = open('./X_test.txt','r')
# y_train = open('./y_train.txt','r')
# y_test = open('./y_test.txt','r')
# word2vec = Word2Vec.load("word2vec.model")
# def preprocess(X,y):
#     X_list = []
#     y_list = []
#     for line in tqdm(X.readlines()):
#         line = line.strip()
#         line = line.split(',')[1:]
#         X_list.append(line) # lists in list
#     for line in tqdm(y.readlines()):
#         line = line.strip()
#         line = line.split(',')[1:]
#         line = np.array(list(map(int,line)))
#         y_list.append(line)
#     return X_list,y_list
# X_train,y_train = preprocess(X_train,y_train)
# X_test,y_test = preprocess(X_test,y_test)

# # with open('./word_id_dict.pkl', 'rb') as fr:
# #     word_id_dict = pkl.load(fr)

# output_dim = len(y_test[0])
# num_words = len(word_id_dict)
# sentence_len = 500

# train_dataset = Text_Dataset(X_train,y_train,num_words=num_words,sentence_len=sentence_len)
# train_loader = DataLoader(train_dataset,batch_size = 3, shuffle=True)
# test_dataset = Text_Dataset(X_test,y_test,num_words=num_words,sentence_len=sentence_len)
# test_loader = DataLoader(test_dataset,batch_size = 1, shuffle=False)

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# model = CAML(num_words=num_words,sentence_len=sentence_len,output_dim=output_dim,embedding_dim=50,hidden_channels=10,kernel_size=3).to(device)
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-04,betas=[0.999,0.9],eps=1e-05)
# os.makedirs("./saved_model",exist_ok=True)
# for epoch in range(300):
#     train_loss = 0

#     model.train()
#     prediction_list = []
#     labels_list = []
#     for i,(X,y) in tqdm(enumerate(train_loader)):

#         optimizer.zero_grad()
#         X = X.to(device)
#         y = y.to(device)
#         prediction = model(X)
#         loss = criterion(prediction,y.float())
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         if len(prediction_list)<100 :
#             prediction_list.append(prediction.cpu().detach().numpy())
#             labels_list.append(y.cpu().detach().numpy())
#         if i % 5000 == 999 :
#             labels_np = np.concatenate(labels_list,axis=0)
#             prediction_np = np.concatenate(prediction_list,axis=0)

#             idx = np.argwhere(np.all(labels_np == 0, axis=0))
#             labels_np = np.delete(labels_np, idx, axis=1)
#             prediction_np = np.delete(prediction_np, idx, axis=1)

#             AUROC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
#             AUROC_micro = roc_auc_score(labels_np,prediction_np,average='micro')
#             AUPRC_macro = average_precision_score(labels_np,prediction_np,average='macro')
#             AUPRC_micro = average_precision_score(labels_np,prediction_np,average='micro')

#             print(AUROC_macro)
#             print(AUROC_micro)
#             print(AUPRC_macro)
#             print(AUPRC_micro)

#             prediction_list = []
#             labels_list = []
#     prediction_list = []
#     labels_list = []
#     model.eval()
#     torch.save({
#         'model_state_dict' : model.state_dict()
#     },'./saved_model/model_'+str(epoch)+'.pth')
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

        

