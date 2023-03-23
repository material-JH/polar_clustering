#%%
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

arr = []
for file in os.listdir('output'):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP_80'):
        print(file)
        arr.append(np.load(f'output/{file}'))

for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
#%%
start_pos = [60, 162]
# start_pos = [12, 113]
# start_pos = [212, 212]
rad = 50

disk = []
for n in range(len(arr)):
    disk.append(arr[n][start_pos[0]:start_pos[0] + rad,
                       start_pos[1]:start_pos[1] + rad])
    plt.imshow(disk[n])
    plt.show()
center_of_mass_position(disk[0])
# %%
np.save('output/disk_011_2.npy', np.array(disk))
# %%
