#%%
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

arr = []
for file in os.listdir('output'):
    if not file.__contains__('array'):
        print(file)
        arr.append(np.load(f'output/{file}'))

for n in range(len(arr)):
    arr[n] = resize(arr[n][0,0],  [r + 30 for r in arr[n][0,0].shape])
start_pos = [7, 101]
rad = 50

disk = []
for n in range(len(arr)):
    disk.append(arr[n][start_pos[0]:start_pos[0] + rad,
                       start_pos[1]:start_pos[1] + rad])
center_of_mass_position(disk[0])
plt.imshow(disk[0])
# %%
np.save('output/disk.npy', np.array(disk))
# %%
