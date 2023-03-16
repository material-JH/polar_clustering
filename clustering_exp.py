#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from main import *

def plot_vertical(data):
    
    for i in range(0, 40, 5):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
        for j in range(5):
            axs[j].imshow(data[j, i, 5])
            axs[j].axis('off')
        plt.show()

def center_of_mass_position(arr):
    rows, cols = arr.shape
    total_mass = arr.sum()
    if total_mass == 0:
        return None
    y_indices, x_indices = np.indices((rows, cols))
    x_c = int((arr * x_indices).sum() / total_mass)
    y_c = int((arr * y_indices).sum() / total_mass)
    return (y_c, x_c)
# matplotlib.use('QtAgg')
#%%
data = load_data(r"/mnt/c/Users/em3-user/Documents/set4")
#%%
data_post = fn_on_resized(data, imutils.rotate, 81)
com = fn_on_resized(data_post, center_of_mass_position)
com = list(map(int, np.sum(com, axis=(0, 1, 2, 3)) / reduce(lambda x, y: x * y, com.shape[:3])))
data_post = shift_n_crop(data_post, int(data.shape[-1] / 3.8),
                        shift_x = com[0] - 256,
                        shift_y = com[1] - 256)

#%%
disk_pos_002 = [5, 102]
disk_pos_011 = [53, 145]
data_post_002 = crop(data_post, 50, disk_pos_002)
data_post_011 = crop(data_post, 50, disk_pos_011)
data_post_002_norm = normalize_Data(data_post_002)
data_post_011_norm = normalize_Data(data_post_011)
#%%
n_neighbors = 15
n_components = 2
min_dist = 0.2

embedding, labels = get_emb_lbl(data_post_002)


#%%

from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=3, affinity='rbf', assign_labels='kmeans')
labels = spectral.fit_predict(embedding)
# embedding, labels = get_emb_lbl(data_post, n_neighbors, n_components, min_dist)

plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], c='blue', label='Cluster 1')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], c='red', label='Cluster 2')
plt.scatter(embedding[labels == 2, 0], embedding[labels == 2, 1], c='green', label='Cluster 3')

plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
for n, i in enumerate(labels.reshape(data_post_002.shape[:3])):
    axs[n].imshow(i)
    axs[n].axis('off')
plt.show()

#%%
xyz = reduce((lambda x, y: x * y), data.shape[:3])
sel_dat = data_post_011_norm.reshape(xyz , -1)[labels == 0]
reducer = umap.UMAP()
embedding2 = reducer.fit_transform(sel_dat)
spectral.n_clusters = 2
labels2 = spectral.fit_predict(embedding2)
plt.scatter(embedding2[labels2 == 0, 0], embedding2[labels2 == 0, 1], c='blue', label='Cluster 1')
plt.scatter(embedding2[labels2 == 1, 0], embedding2[labels2 == 1, 1], c='red', label='Cluster 2')

# embedding2, labels2 = get_emb_lbl(sel_dat, n_components, n_neighbors, min_dist)
#%%
new_lbl = []
n = 0
for l in labels:
    if l == 0:
        new_lbl.append(labels2[n] * 3)
        n += 1
    else:
        new_lbl.append(l)
        
lbl_reshape = np.reshape(new_lbl, data_post.shape[:3])
#%%
for i, j in zip(lbl_reshape, data.sum(axis=(-1, -2))):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axs[0].imshow(i)
    axs[1].imshow(j)
    plt.show()
# %%
