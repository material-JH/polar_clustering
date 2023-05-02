#%%
import random
import numpy as np
import os
from lib.main import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from glob import glob
#%%
from tqdm import tqdm
arr = []
fnames = []

circle = get_circle_conv(45)
for file in tqdm(glob('output/dft/*.npy')):
    arr.append(np.load(file).astype(np.float16))
    file = file.rsplit('/', 1)[-1]
    fnames.append(file[:-4])
for n in range(len(arr)):
    arr[n] = resize(arr[n][0,0],  [r + 120 for r in arr[n][0,0].shape])
    # arr[n] = arr[n]np.roll(arr[n], get_center(), axis=0)
#%%
disk_011 = [62, 159]
disk_002 = [52, 150]
start_pos_m002 = [212, 113]
arr = np.array(arr)
rad = 50
#%%
plot_tk(arr)
#%%
# arr = fn_on_resized(arr, normalize_Data)
arr[np.where(arr > 3)] = 3
plot_tk(arr)
#%%
disk = {}
for n in range(len(arr)):
    name = fnames[n]
    if abs(float(name.split('_')[2])) > 0.1:
        continue
    img = crop(arr[n], rad, disk_002)
    # img2 = crop(arr[n][::-1], rad, disk_002)
    # name_sp = name.split('_')
    # name_sp[2] = str(-float(name_sp[2]))
    # name2 = '_'.join(name_sp)
    disk[name] = img
    # disk[name2] = img2

#%%
name_sp = name.split('_')
name_sp[2] = str(-float(name_sp[2]))
name = '_'.join(name_sp)

p = [float(name.split('_')[2]) for name in fnames if abs(float(name.split('_')[2])) < 0.1]
p = list(set(p))
plt.plot(p)
# %%
np.savez('output/disk_002_dft.npz', **disk)
print('saved')

#%%

disk_all = np.load('output/disk_002_dft.npz')
disk = []
for k, v in disk_all.items():
    disk.append(v)
disk = np.array(disk)
disk = fn_on_resized(disk, normalize_Data)
# disk[np.where(disk > 3)] = 3
# disk[:,0] = 8
# plot_tk(disk)
# %%
plot_tk(disk)
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
