import imutils
import hyperspy.api as hs
import os
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

# import scipy.signal as sig
try:
    import cupyx.scipy.signal as sig
    import cupy as cp
except:
    import scipy.signal as sig

def get_circle_conv(size):
    center = (size // 2, size // 2)  # Center point of the circle
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    circle = np.zeros((size, size))
    circle[dist <= center[0]] = 1
    return circle

def plot_vertical(data):
    
    for i in range(0, 40, 5):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
        for j in range(5):
            axs[j].imshow(data[j, i, 5])
            axs[j].axis('off')
        plt.show()

def get_center(arr, conv):
    # if cp.cuda.runtime.
    try:
        arr = cp.asarray(arr)
        conv = cp.asarray(conv)
    except:
        pass
    # result = sig.convolve2d(data_post_011_norm[0,0,0], circle, mode='same')
    result = sig.convolve2d(arr, conv, mode='same')
    # Find the maximum position
    max_pos = np.unravel_index(np.argmax(result.get()), result.shape)

    return (max_pos[0], max_pos[1])

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
        if i.__contains__('dm'):
            data.append(hs.load(os.path.join(path, i)).data)
    data = np.stack(data, axis=0)
    data = data.swapaxes(1, 3).swapaxes(2, 4)
    
    return data

def _crop_from_center(data, size, com, i):
    pos_x, pos_y = com[i]
    return data[pos_y - size // 2: pos_y + size // 2,
                pos_x - size // 2: pos_x + size // 2]

def crop_from_center(data, size, com):
    return fn_on_resized(data, _crop_from_center, size, com, list=True)
                
def _crop(data, size, position):
    return data[position[0]:position[0] + size, 
                position[1]:position[1] + size]

def crop(data, size, position):
    return fn_on_resized(data, _crop, size, position)

def rotate_by_cen(data, angle, com, i):
    return imutils.rotate(data, angle, center=com[i])

def fn_on_resized(data, fn, *args, **kwargs):
    shape = data.shape
    tmp = np.reshape(data, (-1, *shape[-2:]))
    output = []
    for i in range(tmp.shape[0]):
        if 'list' in kwargs.keys():
            output.append(fn(tmp[i], *args, i))
        else:
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