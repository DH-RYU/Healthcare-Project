import numpy as np
import pandas as pd
import os
import pickle as pkl
from tqdm.auto import tqdm
import re

curr_dir = os.getcwd()
# noteevents = pd.read_csv(os.path.join(curr_dir,'filtered_noteevents.csv.gz'))
# word_bag = set()

# p = re.compile("[a-zA-Z'0-9]+")
# for idx,row in tqdm(noteevents.iterrows()):
#     hadm_id = row.HADM_ID
#     text = row.TEXT
#     if not((hadm_id % 10 == 8) or (hadm_id % 10 == 9)) :
#         text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text).lower()
#         text_list = p.findall(text)
#         word_bag.update(text_list)

# num_words = len(word_bag)
# word_id_dict = dict(zip(word_bag,list(range(num_words))))

# # save data
# with open('./word_id_dict_0.pkl','wb') as fw:
#     pkl.dump(word_id_dict, fw)
'''
'''

# load data
with open('./word_id_dict_0.pkl', 'rb') as fr:
    word_id_dict = pkl.load(fr)
print(len(word_id_dict))
noteevents = pd.read_csv(os.path.join(curr_dir,'filtered_noteevents.csv.gz'))
p = re.compile("[a-zA-Z'0-9]+")

## Delete 
# id_word_dict = dict([(value, key) for key, value in word_id_dict.items()])
# id_word_dict[len(word_id_dict)+1] = "PAD"
# id_word_dict[len(word_id_dict)] = "UKN"
## Delete 

def word_to_id(x,dictionary):
    if x in dictionary.keys():
        return str(dictionary[x])
    else :
        return str(len(dictionary)) # Unknown word

X_train = open('./X_train_tmp.txt','w')
X_test = open('./X_test_tmp.txt','w')
y_train = open('./y_train.txt','r')
y_test = open('./y_test.txt','r')
count = 0
for line in tqdm(y_train.readlines()):
    hadm_id = int(line.split(",")[0])
    text = noteevents[noteevents.HADM_ID==hadm_id].TEXT.iloc[0]
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = p.findall(text.lower())
    text = list(map(lambda x : word_to_id(x,word_id_dict),text))
    # Delete
    # text_reverse = list(map(lambda x : id_word_dict[int(x)],text))
    # count += 1
    # if count == 10 :
        # break
    # Delete
    X_train.write(str(hadm_id)+","+",".join(text)+'\n')

for line in tqdm(y_test.readlines()):
    hadm_id = int(line.split(",")[0])
    text = noteevents[noteevents.HADM_ID==hadm_id].TEXT.iloc[0]
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = p.findall(text.lower())
    text = list(map(lambda x : word_to_id(x,word_id_dict),text))
    X_test.write(str(hadm_id)+","+",".join(text)+'\n')

X_train.close()
X_test.close()
y_train.close()
y_test.close()