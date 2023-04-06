#%%
import random
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

#%%
from tqdm import tqdm
arr = []
fnames = []
n = 0

circle = get_circle_conv(45)
for file in tqdm(os.listdir('output/all')):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP_a') or file.__contains__('DP_c') or file.__contains__('DP_g'):
        arr.append(np.load(f'output/all/{file}'))
        fnames.append(file)
        n += 1
#%%
for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
    # arr[n] = arr[n]np.roll(arr[n], get_center(), axis=0)
start_pos_011 = [60, 158]
start_pos_002 = [12, 113]
start_pos_m002 = [212, 113]
arr = np.array(arr)
rad = 54
#%%
unique_pre = []
for i in fnames:
    a, b = i.rsplit('_', 1)[0].rsplit('_', 1)
    unique_pre.append('_'.join([a, str(round(int(b), -1))]))
#%%
print(unique_pre)
#%%
aver_arr = np.zeros((len(unique_pre), len(arr[0]), len(arr[0][0])))
num_arr = np.zeros(len(unique_pre))
for dp, name in zip(arr, fnames):
    for i, sup in enumerate(unique_pre):
        if name.startswith(sup):
            aver_arr[i] += dp
            num_arr[i] += 1
    
for i in range(len(aver_arr)):
    aver_arr[i] /= num_arr[i]
#%%
sep = {}
for n, i in enumerate(set([i.split('_')[3] for i in fnames])):
    sep[i] = n

disk = {}
for i in sep.keys():
    disk[i] = []
    
for i in range(4):
    aver_arr = aver_arr[::-1,:]
    if i % 2 == 0:
        aver_arr = aver_arr[:, ::-1]
    if i // 2 == 0:
        start_pos_011[1] = 159
    else:
        start_pos_011[1] = 156
    # for n in range(1):
    for n in range(len(aver_arr)):
        dn = np.sum(crop(aver_arr[n], rad, start_pos_002))
        up = np.sum(crop(aver_arr[n], rad, start_pos_m002))
        
        img = crop(aver_arr[n], rad, start_pos_011)
        name = unique_pre[n]
        disk[name.split('_')[3]].append(img)
    # prime = get_center(disk[0], circle)
    # for i in range(len(disk)):
    #     disk[i] = np.roll(disk[i], prime[0] - get_center(disk[i], circle)[0], axis=0)
    #     disk[i] = np.roll(disk[i], prime[1] - get_center(disk[i], circle)[1], axis=1)
    #     print(get_center(disk[i], circle))
    # print(prime)

for i in sep.keys():
    disk[i] = np.array(disk[i])
    print(len(disk[i]))

# %%
np.savez('output/disk_011_all_2.npz', **disk)

import pickle

with open('output/sep_011_all_2.pkl', 'wb') as f:
    pickle.dump(sep, f)
print('saved')

#%%