import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
import torchvision
from torchvision import models
import os
import numpy as np
from PIL import Image
import pickle as pkl
from tqdm.auto import tqdm

model = models.resnet50(pretrained=True)
X_test = open('./X_test.pkl','rb')
X_test = pkl.load(X_test)
y_test = open('./y_test.txt','r')
output = open('./20213205_model.txt','w')
label_dict = {}
for line in tqdm(y_test.readlines()):
    line = line.strip().split(',')
    study_id = line[0]
    label = line[1:]
    label = np.array(list(map(int,label)))
    label_dict[study_id] = label

class CXR_dataset(Dataset):
    def __init__(self,data,label_dict):
        super(CXR_dataset,self).__init__()
        self.data = data
        self.len = len(data)
        self.label_dict = label_dict
    def __len__(self):
        return self.len

    def __getitem__(self, idx) :
        X,study_id = self.data[idx]
        X = torch.from_numpy(X).unsqueeze(0)
        X = X.repeat(3,1,1)
        y = torch.from_numpy(self.label_dict[study_id])
        return X,y

num_classes = 14
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device('cuda:1')
model.to(device)

dataset = CXR_dataset(X_test,label_dict)
loader = DataLoader(dataset,batch_size = 20,shuffle=True)
criterion = nn.BCELoss()

labels_list = []
prediction_list = []
total_loss = 0

checkpoint = torch.load('./model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
for i,(X,y) in tqdm(enumerate(loader)):
    X = X.to(device).float()
    y = y.to(device)
    pred = model(X)
    prediction_list.append(torch.sigmoid(pred).cpu().detach().numpy())
    labels_list.append(y.cpu().detach().numpy())
labels_np = np.concatenate(labels_list,axis=0)
prediction_np = np.concatenate(prediction_list,axis=0)

AUROC_macro = roc_auc_score(labels_np,prediction_np,average='macro')
AUROC_micro = roc_auc_score(labels_np,prediction_np,average='micro')
AUPRC_macro = average_precision_score(labels_np,prediction_np,average='macro')
AUPRC_micro = average_precision_score(labels_np,prediction_np,average='micro')
output.write('20213205\n')
output.write(str(AUROC_macro)+'\n')
output.write(str(AUROC_micro)+'\n')
output.write(str(AUPRC_macro)+'\n')
output.write(str(AUPRC_micro)+'\n')
