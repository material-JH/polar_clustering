#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
from lib.plot import plot_tk
import cv2
import gc

#%%
# data = np.load('/home/jinho93/project/Fe2O3/18_64.npy')
data = load_data(r"/home/jinho93/project/BST/data/set4_SRO")
data = crop(data, 450, [50, 50])
# data = np.swapaxes(data, 0, 2)
# data = np.swapaxes(data, 1, 3)
#%%
vmin = np.min(data)
vmax = np.max(data)
fig, ax = plt.subplots(1, 5)
for i in range(5):
    ax[i].imshow(data[i].sum(axis=(-1, -2)), cmap='RdBu', vmin=vmin, vmax=vmax)
    ax[i].axis('off')
    ax[i].set_title(f'{i - 2}V')

#%%
plot_tk(np.reshape(data, (-1, *data.shape[-2:])), vmax=np.max(data) / 4, vmin=np.min(data))
#%%

around_center = crop(data, 125, [150, 150])
circle = get_circle_conv(42)
com = []
for i in np.reshape(around_center, (-1, *around_center.shape[-2:])):
    com.append(get_center(i, circle))

for i in range(len(com)):
    com[i] = [com[i][0] + 150, com[i][1] + 150]

#%%
data_post = fn_on_resized(data, rotate_by_cen, 82, com, list=True)
data_post = np.reshape(data_post, (-1, *data_post.shape[-2:]))
tmp = []
for i in range(len(data_post)):
    tmp.append(crop_from_center(data_post[i], 250, [com[i]]))
#%%
data_post = np.reshape(tmp, (*data.shape[:-2], *tmp[0].shape))
#%%
plot_tk(np.reshape(data_post, (-1, *data_post.shape[-2:])), vmax=np.max(data_post) / 4, vmin=np.min(data_post))
#%%
plt.imshow(data_post[0,15,0])
plt.plot([125, 125], [0, 250], 'r')
plt.plot([0, 250], [125, 125], 'r')
#%%
disk_pos_002 = [9, 99]
disk_pos_m002 = [193, 99]
disk_pos_011 = [55, 145]
disk_pos_m011 = [147, 55]

data_post_002 = crop(data_post, 50, disk_pos_002)
data_post_m002 = crop(data_post, 50, disk_pos_m002)
data_post_011 = crop(data_post, 50, disk_pos_011)
data_post_m011 = crop(data_post, 50, disk_pos_m011)
n = 5
data_post_002_norm = fn_on_resized(data_post_002, cv2.GaussianBlur, (n, n), 0)
data_post_m002_norm = fn_on_resized(data_post_m002, cv2.GaussianBlur, (n, n), 0)
data_post_011_norm = fn_on_resized(data_post_011, cv2.GaussianBlur, (n, n), 0)
data_post_m011_norm = fn_on_resized(data_post_m011, cv2.GaussianBlur, (n, n), 0)
#%%
fig, ax = plt.subplots(1,5)
# vmax = np.max(output)
# vmin = np.min(output)
# output[lbl==0] = 0
sumI_002 = data_post_002.sum(axis=(-1, -2))
sumI_002 = sumI_002 / sumI_002.max()
sumI_m002 = data_post_m002.sum(axis=(-1, -2))
sumI_m002 = sumI_m002 / sumI_m002.max()
sumI_011 = data_post_011.sum(axis=(-1, -2))
sumI_011 = sumI_011 / sumI_011.max()


sumI_m002 = data_post_m002.sum(axis=(-1, -2))

imgs = np.stack([sumI_m002, np.zeros_like(sumI_m002), sumI_002])
imgs = imgs.swapaxes(0, 1)
vmax = np.max(imgs)
vmin = np.min(imgs)
for n, img in enumerate(sumI_m002):
    # if n % 2 == 1:
    #     continue
    # n = n // 2
    # pcm = ax[n].imshow(img, cmap='RdBu', interpolation='bessel',vmin=vmin,vmax=vmax)
    pcm = ax[n].imshow(img, cmap='RdBu_r',vmin=vmin,vmax=vmax, alpha=.6, aspect='auto')
    ax[n].set_title(f'{n - 2}V')

    ax[n].axis('off')
# plt.colorbar(pcm, ax=ax[-2])
plt.show()


#%%
plot_tk(np.reshape(data_post_m011_norm, (-1, *data_post_002_norm.shape[-2:])))
#%%
plt.imshow(data_post_002_norm[2,25,2])
#%%
np.save('output/set3_SRO_002.npy', data_post_002_norm)
np.save('output/set3_SRO_m002.npy', data_post_m002_norm)
# np.save('output/set2_SRO_large.npy', small)
# np.save('output/set4_SRO_002.npy', data_post_002_norm)
# np.save('output/set4_SRO_m002.npy', data_post_m002_norm)
# np.save('output/set1_Ru_011.npy', data_post_011_norm)
# np.save('output/set1_Ru_m011.npy', data_post_m011_norm)
# np.save('output/set1_SRO_011.npy', data_post_011_norm)
# %%
plt.imshow(data_post[0, 15, 0] - data_post[0, 35, 4], vmax=data_post[0, 15, 0].max(), vmin=data_post[0, 15, 0].min())
# %%
plt.imshow(imutils.resize(data_post[0, 15, 0], 100))
# %%

data_post_002_norm = data_post_002_norm - data_post_002_norm.mean()
data_post_011_norm = data_post_011_norm - data_post_011_norm.mean()
data_post_002_norm = data_post_002_norm / data_post_002_norm.std()
data_post_011_norm = data_post_011_norm / data_post_011_norm.std()

means = []
stds = []
for i in range(data_post_011_norm.shape[1]):
    means.append(data_post_002_norm[0, i, 0].mean())
    stds.append(data_post_002_norm[0, i, 0].std())
fig, ax = plt.subplots()
ax.errorbar(range(data_post_011_norm.shape[1]), means, yerr=stds, fmt='o', capsize=5)
# %%
