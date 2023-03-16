#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from main import *

# matplotlib.use('QtAgg')

data = load_data(r"/mnt/c/Users/em3-user/Documents/set4")
#%%
# data_post = fn_on_resized(data, imutils.rotate, 83)
data_post = fn_on_resized(data, imutils.rotate, 82)
data_post = shift_n_crop(data_post, 
                       crop_amount = int(data.shape[-1] / 3.8),
                           shift_x = -2,
                        #    shift_x = -6,
                           shift_y = 5)

# pos = [51, 142]
pos = [5, 96]
data_post = crop(data_post, 50, pos)
data_post = normalize_Data(data_post)

for i in range(0, 40, 5):
    plt.imshow(data_post[0, i, 5])
    plt.axis('off')
    plt.show()

#%%
n_neighbors = 100
n_components = 2
min_dist = 0.2

embedding, labels = get_emb_lbl(data_post, n_neighbors, n_components, min_dist)

plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], c='blue', label='Cluster 1')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], c='red', label='Cluster 2')

plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()
#%%

for i in labels.reshape(data_post.shape[:3]):
    plt.imshow(i)
    plt.show()

#%%
xyz = reduce((lambda x, y: x * y), data.shape[:3])
sel_dat = data_post.reshape(xyz , -1)[labels == 1]
embedding2, labels2 = get_emb_lbl(sel_dat, n_neighbors, n_components, min_dist)
#%%
new_lbl = []
n = 0
for l in labels:
    if l == 1:
        new_lbl.append(labels2[n] + 1)
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
