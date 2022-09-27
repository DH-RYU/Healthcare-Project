import os
import pandas as pd
from tqdm.auto import tqdm

f = open('./y_test.txt','r')
count = 0
line_count = 0
for line in tqdm(f.readlines()):
    data = line.strip().split(',')
    data = list(map(int,data))
    count += data.count(1)
    line_count += 1

print(line_count)
print(count)

f = open('./y_train.txt','r')
count = 0
line_count = 0
for line in tqdm(f.readlines()):
    data = line.strip().split(',')
    data = list(map(int,data))
    count += data.count(1)
    line_count += 1

print(line_count)
print(count)