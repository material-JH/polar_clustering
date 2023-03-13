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

# matplotlib.use('Qt5Agg')

data = []
os.chdir(r"/mnt/c/Users/em3-user/Documents/set4")
for i in os.listdir():
    # path = r"C:/Users/em3-user/Documents/set1/-2.dm4"
    data.append(hs.load(i).data)
data = np.stack(data, axis=0)
#%%

data = data.swapaxes(1, 3).swapaxes(2, 4)
#%%
nrx, nry, nkx, nky = data.shape[1:]

crop_amount = nkx // 5 * 2
crop_dat = np.zeros((*data.shape[:3], nkx - 2 * crop_amount, nky // 2))

for i in range(data.shape[0]):
    for j in range(nrx):
        for k in range(nry):
            crop_dat[i, j, k] = NormalizeData(data[i, j, k, 
                                 crop_amount:nkx - crop_amount,
                                 crop_amount:nkx - crop_amount])

print(crop_dat.shape)
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

data = []
for i in os.listdir():
    data.append(np.load(i))
# %%
