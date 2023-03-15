#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import imutils as imutils
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from main import *
import hyperspy.api as hs

# matplotlib.use('QtAgg')

data = []
os.chdir(r"/mnt/c/Users/em3-user/Documents/set4")
for i in os.listdir():
    # path = r"C:/Users/em3-user/Documents/set1/-2.dm4"
    data.append(hs.load(i).data)
data = np.stack(data, axis=0)
data = data.swapaxes(1, 3).swapaxes(2, 4)
#%%
angle = 82
nrx, nry, nkx, nky = data.shape[1:]

crop_amount = int(nkx / 3.8)
shift_x = -2
shift_y = -12
crop_dat = np.zeros((*data.shape[:3], nkx - 2 * crop_amount, nkx - 2 * crop_amount))
for i in range(data.shape[0]):
    for j in range(nrx):
        for k in range(nry):
            rotate = imutils.rotate(data[i, j, k], angle)
            crop_dat[i, j, k] = (rotate[ 
                                 crop_amount + shift_x:nkx - crop_amount + shift_x,
                                 crop_amount + shift_y:nkx - crop_amount + shift_y])
#%%
print(crop_dat.shape)
# pos = [53 + shift_x, 154 + shift_y]
pos = [5, 96]
size = 50
crop_dat2 = np.zeros((*crop_dat.shape[:3], size, size))
for i in range(crop_dat.shape[0]):
    for j in range(nrx):
        for k in range(nry):
            crop_dat2[i, j, k] = crop_dat[i, j, k][pos[0]:pos[0] + size, 
                                         pos[1]:pos[1] + size]
for i in range(0, 30, 5):
    # plt.imshow(crop_dat[0, i, 5])
    # plt.plot([crop_dat.shape[3] / 2] * 2 , [0, crop_dat.shape[3]])
    # plt.plot([0, crop_dat.shape[3]], [crop_dat.shape[3] / 2] * 2 )
    # plt.axis('off')
    plt.show()
    plt.imshow(crop_dat2[0, i, 5])
    plt.axis('off')
    plt.show()

#%%

fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(4, 10))

# Plot each image on a subplot
for i in range(10):
    for j in range(4):
        axes[i, j].imshow(NormalizeData(crop_dat2[0, i * 4, j * 2]))

for ax in axes.flatten():
    ax.axis('off')
    
plt.show()
#%%

plt.imshow(crop_dat2[0].sum(axis=(2, 3)))
plt.axis('off')

#%%
crop_dat_r = crop_dat.reshape(-1, nkx // 2, nky // 2)
print(crop_dat_r.shape)
#%%
# Choose UMAP parameterss
n_neighbors = 100
n_components = 3
min_dist = 0.1

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(crop_dat_r.reshape(crop_dat_r.shape[0], -1))

from sklearn.cluster import KMeans

# Assuming `embedding` contains the reduced dimensional representation of your data
kmeans = KMeans(n_clusters=2)
kmeans.fit(embedding)

# Assign cluster labels to each data point
labels = kmeans.labels_
import matplotlib.pyplot as plt

# Assuming `embedding` and `labels` have been obtained as shown in the previous code snippet
# Plot the scatter plot with the first cluster in blue and the second cluster in red
plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], c='blue', label='Cluster 1')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], c='red', label='Cluster 2')

plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()

# %%
os.chdir('/mnt/c/Users/em3-user/Documents/GitHub/polar_clustering/output')

data2 = []
for i in os.listdir():
    data2.append(np.load(i))
# %%


plt.imshow(data2[0][0,0])


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4, 10))

# Plot each image on a subplot
for i in range(3):
    for j in range(2):
        axes[i, j].imshow(NormalizeData(data2[i * j][0,0]))

for ax in axes.flatten():
    ax.axis('off')
    
plt.show()
# %%
