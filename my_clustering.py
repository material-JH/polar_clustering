#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from collections import Counter
from matplotlib import cm
from PIL import Image
import imutils as imutils
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
matplotlib.use('Qt5Agg')



def one_round_clustering(n_clusters, manifold_data):
    if np.shape(manifold_data)[1] > 1000:
        manifold_clustering_result = MiniBatchKMeans(n_clusters=n_clusters).fit(manifold_data)
    else:
        manifold_clustering_result = KMeans(n_clusters=n_clusters).fit(manifold_data)

    labels = manifold_clustering_result.labels_ + 1

    return labels, manifold_clustering_result.cluster_centers_


def get_rotation_matrix(i_v, unit=None):
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    i_v /= np.linalg.norm(i_v)
    # Get axis
    uvw = np.cross(i_v, unit)
    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)
    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw
    # Compute rotation matrix - re-expressed to show structure
    return (
            rcos * np.eye(3) +
            rsin * np.array([
        [0, -w, v],
        [w, 0, -u],
        [-v, u, 0]
    ]) +
            (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

def NormalizeData(data):
    return (data - np.mean(data)) / data.


X = []
for i in os.listdir('output'):
    X.append(np.load(os.path.join('output', i))[0,0])
X = np.stack(X, axis=0)


X = X.reshape((-1, X.shape[1] * X.shape[2]))

#%%
# Choose UMAP parameters
n_neighbors = 100
n_components = 3
min_dist = 0.1

embedding = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, random_state=42).fit_transform(X)

# Visualize the results
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=10)
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("UMAP projection of MNIST dataset")
# plt.show()# %%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:,0],embedding[:,1],embedding[:,2])
plt.show()
# %%
