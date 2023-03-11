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
    return (data - np.mean(data)) / np.std(data)


X = []
for i in os.listdir('output'):
    X.append(np.load(os.path.join('output', i))[0,0])
X = np.stack(X, axis=0)


X = X.reshape((-1, X.shape[1] * X.shape[2]))

X = NormalizeData(X)

#%%
# Choose UMAP parameters
n_neighbors = 100
n_components = 3
min_dist = 0.1

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(X)

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
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

# Assuming `X` is a numpy array with your data
# Compute the UMAP embedding
umap_model = umap.UMAP()
embedding = umap_model.fit_transform(X)

# Compute the K-means clusters
kmeans_model = KMeans(n_clusters=2)
labels = kmeans_model.fit_predict(X)

# Compute the pairwise distances between the data points
distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

# Plot the scatter plot with the K-means clusters and scaled markers based on the distances
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels,
            s=1 + distances.flatten() / np.max(distances) * 100)
plt.colorbar()
plt.title('UMAP embedding with K-means clustering and scaled markers based on distances')
plt.show()


# %%
distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

# Compute the mean distances for each cluster
mean_distances = []
for i in range(len(np.unique(labels))):
    mean_distance = np.mean(distances[labels == i])
    mean_distances.append(mean_distance)

scaled_sizes = 1 + distances.flatten() / np.max(mean_distances) * 100
for i in range(len(np.unique(labels))):
    scaled_sizes[labels == i] /= mean_distances[i]

print(scaled_sizes.shape)
# %%
plt.plot(distances[:,0])
plt.show()
# %%
