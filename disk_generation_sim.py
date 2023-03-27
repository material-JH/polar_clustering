#%%
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

arr = []
n = 0
for file in os.listdir('output'):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP'):
        arr.append(np.load(f'output/{file}'))
        n += 1
for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
start_pos = [60, 158]
# start_pos = [12, 113]
# start_pos = [0, 0]
rad = 50

disk = []
for i in range(4):
    if i % 2 == 0:
        for n in range(len(arr)):
            arr[n] = arr[n][:, ::-1]
    for n in range(len(arr)):
        arr[n] = arr[n][::-1, :]
    for n in range(len(arr)):
        disk.append(arr[n][start_pos[0]:start_pos[0] + rad,
                        start_pos[1]:start_pos[1] + rad])
        if n % 20 == 0:
            plt.imshow(disk[n])
            plt.axis('off')
            plt.show()
# %%
np.save('output/disk_011_4.npy', np.array(disk))
# %%

fig, ax = plt.subplots(2, 5)
for i in range(5):
    for j in range(2):
        ax[j, i].imshow(disk[i + j * 5])
        ax[j, i].axis('off')
#%%
plt.imshow(arr[0])
# %%
