#%%
import numpy as np
import matplotlib.pyplot as plt
from main import *
import cv2
import gc

#%%
data = np.load('/home/jinho93/project/Fe2O3/18_64.npy')
data = np.swapaxes(data, 0, 2)
data = np.swapaxes(data, 1, 3)

#%%
import sys
print(sys.getsizeof(data))

#%%
plot_tk(np.reshape(data, (-1, *data.shape[-2:])))
#%%
data = load_data(r"/home/jinho93/project/Fe2O3")
# data = crop(data, 450, [50, 50])
#%%
data[1] = np.swapaxes(data[1], 0, 2)
data[1] = np.swapaxes(data[1], 1, 3)
#%%
plot_tk(np.reshape(data[1], (-1, *data[1].shape[-2:])))
#%%
np.save('output/Fe2O3_old.npy', data[0])
#%%

around_center = crop(data, 125, [150, 150])
circle = get_circle_conv(42)
com = []
for i in np.reshape(around_center, (-1, *around_center.shape[-2:])):
    com.append(get_center(i, circle))

for i in range(len(com)):
    com[i] = [com[i][0] + 150, com[i][1] + 150]

#%%
j = 40
plt.imshow(data[0,j,0])

plt.plot([com[j * 10][0], com[j * 10][0]], [0, data.shape[-2]], 'r')
plt.plot([0, data.shape[-1]], [com[150][1], com[150][1]], 'r')
plt.xlim([data.shape[-1] // 3, data.shape[-1] // 3 * 2])
plt.ylim([data.shape[-2] // 3 * 2, data.shape[-2] // 3])

#%%
plt.imshow(crop_from_center(data_post[0,10,0], 250, [com[100]]))

#%%
data_post = fn_on_resized(data, rotate_by_cen, 82, com, list=True)
data_post = np.reshape(data_post, (-1, *data_post.shape[-2:]))
tmp = []
for i in range(len(data_post)):
    tmp.append(crop_from_center(data_post[i], 250, [com[i]]))
#%%
data_post = np.reshape(tmp, (*data.shape[:-2], *tmp[0].shape))
#%%
data_post = normalize_Data(data_post)

#%%
plt.imshow(data_post[0,15,0])
plt.plot([125, 125], [0, 250], 'r')
plt.plot([0, 250], [125, 125], 'r')
#%%
disk_pos_002 = [9, 99]
disk_pos_011 = [55, 145]
data_post_002 = crop(data_post, 50, disk_pos_002)
data_post_011 = crop(data_post, 50, disk_pos_011)
# data_post_002_norm = normalize_Data(data_post_002)
# data_post_011_norm = normalize_Data(data_post_011)
n = 5
data_post_002_norm = fn_on_resized(data_post_002, cv2.GaussianBlur, (n, n), 0)
data_post_011_norm = fn_on_resized(data_post_011, cv2.GaussianBlur, (n, n), 0)

#%%
plot_tk(np.reshape(data_post_002_norm, (-1, *data_post_002_norm.shape[-2:])))
#%%
plt.imshow(data_post_002_norm[2,25,2])
#%%
plot_tk(np.reshape(data_post_002_norm, (-1, *data_post_002_norm.shape[-2:])))
#%%
# np.save('output/set3_Ru_002.npy', data_post_002_norm)
# np.save('output/set2_SRO_large.npy', small)
np.save('output/set1_SRO_002.npy', data_post_002_norm)
np.save('output/set1_SRO_011.npy', data_post_011_norm)
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
