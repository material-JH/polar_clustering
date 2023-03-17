import imutils
import hyperspy.api as hs
import os
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

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

def one_round_clustering(n_clusters, manifold_data):
    if np.shape(manifold_data)[1] > 1000:
        manifold_clustering_result = MiniBatchKMeans(n_clusters=n_clusters).fit(manifold_data)
    else:
        manifold_clustering_result = KMeans(n_clusters=n_clusters).fit(manifold_data)

    labels = manifold_clustering_result.labels_ + 1

    return labels, manifold_clustering_result.cluster_centers_


def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
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


def normalize_Data(data):
    return fn_on_resized(data, _normalize_Data)


def _normalize_Data(data):
    return (data - np.mean(data)) / np.std(data)


def load_data(path):
    data = []
    for i in os.listdir(path):
        data.append(hs.load(os.path.join(path, i)).data)
    data = np.stack(data, axis=0)
    data = data.swapaxes(1, 3).swapaxes(2, 4)
    
    return data

def _shift_n_crop(data, crop_amount, shift_x, shift_y):
    nkx = data.shape[0]
    return data[crop_amount + shift_x:nkx - crop_amount + shift_x,
                crop_amount + shift_y:nkx - crop_amount + shift_y]

def shift_n_crop(data, crop_amount, shift_x, shift_y):
    return fn_on_resized(data, _shift_n_crop, crop_amount, shift_x, shift_y)
                
def _crop(data, size, position):
    return data[position[0]:position[0] + size, 
                position[1]:position[1] + size]

def crop(data, size, position):
    return fn_on_resized(data, _crop, size, position)

def fn_on_resized(data, fn, *args):
    shape = data.shape
    tmp = np.reshape(data, (-1, *shape[-2:]))
    output = []
    for i in range(tmp.shape[0]):
        output.append(fn(tmp[i], *args))
        
    output = np.array(output)
    output = np.reshape(output, (*shape[:3], -1))

    return np.reshape(output, (*shape[:3], int(np.sqrt(output.shape[-1])), -1))

import umap
from functools import reduce


def get_emb_lbl_real(data,n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components= n_components,
                        n_neighbors = n_neighbors,
                        min_dist = min_dist)

    xyz = reduce((lambda x, y: x * y), data.shape[:3])

    embedding = reducer.fit_transform(data.reshape(xyz , -1))
    # Assuming `embedding` contains the reduced dimensional representation of your data
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(embedding)
    # Assign cluster labels to each data point
    labels = kmeans.labels_
    return embedding, labels

def get_emb_lbl(data,n_components=2, n_neighbors=15, min_dist=0.1, n_clusters=2):
    reducer = umap.UMAP(n_components= n_components,
                        n_neighbors = n_neighbors,
                        min_dist = min_dist)

    embedding = reducer.fit_transform(data)
    # Assuming `embedding` contains the reduced dimensional representation of your data
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', assign_labels='kmeans')
    labels = spectral.fit_predict(embedding)

    return embedding, labels