import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

y_train = open('./y_train.txt','r')
y_test = open('./y_test.txt','r')

total_data = 0
train_data = 0
test_data = 0
train_positive_label = 0
test_positive_label = 0

for line in tqdm(y_train.readlines()):
    total_data += 1
    train_data += 1
    line = list(map(int,line.split(',')))
    train_positive_label += sum(line[1:])

for line in tqdm(y_test.readlines()):
    total_data += 1
    test_data += 1
    line = list(map(int,line.split(',')))
    test_positive_label += sum(line[1:])

print('total_data : %d' % total_data)
print('train_data : %d' % train_data)
print('test_data : %d' % test_data)
print('train_positive_data : %d' % train_positive_label)
print('test_positive_data : %d' % test_positive_label)