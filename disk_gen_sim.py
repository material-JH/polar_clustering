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
for file in tqdm(os.listdir('output/dps')):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP_a') or file.__contains__('DP_c') or file.__contains__('DP_g'):
        arr.append(np.load(f'output/dps/{file}').astype(np.float32))
        fnames.append(file)
        n += 1
for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
    # arr[n] = arr[n]np.roll(arr[n], get_center(), axis=0)
start_pos_011 = [58, 158]
start_pos_002 = [12, 113]
start_pos_m002 = [212, 113]
arr = np.array(arr)
rad = 56

#%%
start_pos_001 = [60, 110]
plt.imshow(arr[1111], vmax=1e-4)
# plt.imshow(crop(arr[3], rad, start_pos_001))


#%%
sep = {}
for n, i in enumerate(set([i.split('_')[3] for i in fnames])):
    sep[i] = n

disk = {}
for i in sep.keys():
    disk[i] = []
    
for i in range(4):
    arr = arr[:,::-1,:]
    if i % 2 == 0:
        arr = arr[:,:, ::-1]
    # for n in range(1):
    for n in range(len(arr)):
        # dn = np.sum(crop(arr[n], rad, start_pos_002))
        # up = np.sum(crop(arr[n], rad, start_pos_m002))
        
        img = crop(arr[n], rad, start_pos_011)
        name = fnames[n]
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
np.savez('output/disk_011_5.npz', **disk)

import pickle

with open('output/sep_011_5.pkl', 'wb') as f:
    pickle.dump(sep, f)
print('saved')

#%%

disk_all = np.load('output/disk_011_5.npz')
disk = []
for k, v in disk_all.items():
    disk.extend(v)
disk = np.array(disk)
plot_tk(disk)
# %%
arr = fn_on_resized(arr, normalize_Data)

#%%
arr[arr > 2] = 2
plot_tk(arr)
# %%
