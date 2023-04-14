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
    arr.append(np.load(f'output/dps/{file}').astype(np.float16))
    fnames.append(file[:-4])
    n += 1
for n in range(len(arr)):
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
    # arr[n] = arr[n]np.roll(arr[n], get_center(), axis=0)
disk_011 = [62, 159]
disk_002 = [14, 111]
start_pos_m002 = [212, 113]
arr = np.array(arr)
rad = 50

#%%
num_sep = 2
sep = {}
for n, i in enumerate(set([i.split('_')[num_sep] for i in fnames])):
    sep[i] = n

disk = {}
for i in sep.keys():
    disk[i] = []
    
for n in range(len(arr)):
    img = crop(arr[n], rad, disk_011)
    name = fnames[n]
    disk[name.split('_')[num_sep]].append(img)

for i in sep.keys():
    disk[i] = np.array(disk[i])
    print(len(disk[i]))

# %%
np.savez('output/disk_011.npz', **disk)
print('saved')

#%%

disk_all = np.load('output/disk_011.npz')
disk = []
for k, v in disk_all.items():
    disk.extend(v)
disk = np.array(disk)
plot_tk(disk)
# %%
plot_tk(arr)
# %%
x = []
y = []
for k, v in disk.items():
    x.append(float(k))
    y.append(np.sum(v))

plt.scatter(x, y)
# %%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(disk['-0.43303'][0])
ax[1].imshow(disk['0.43303'][0])
plt.show()
# %%
