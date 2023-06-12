import random
import imutils
import hyperspy.api as hs
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from functools import reduce

try:
    from cuml.manifold import UMAP
    from cuml.cluster import DBSCAN 
    import cupyx.scipy.signal as sig
    import cupy as cp
    print('gpu enabled')
except:
    from sklearn.manifold import UMAP
    from sklearn.cluster import DBSCAN
    import scipy.signal as sig
    print('gpu disabled')


def get_circle_conv(size):
    center = (size // 2, size // 2)  # Center point of the circle
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    circle = np.zeros((size, size))
    circle[dist <= center[0]] = 1
    return circle

def plot_vertical(data, figsize=(8, 12)):
    shape = data.shape
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize)
    for i in range(shape[0]):
        for j in range(shape[1]):
            axs[i, j].imshow(data[i, j])
            axs[i, j].axis('off')
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
    return list(map(int, (max_pos[1], max_pos[0])))


def normalize_Data(data):
    return fn_on_resized(data, _normalize_Data)


def _normalize_Data(data):
    return (data - np.mean(data)) / np.std(data)


def load_data(path):
    data = []
    if os.listdir(path)[0].__contains__('.dm3'):
        postfix = '.dm3'
    else:
        postfix = '.dm4'
    for i in range(-2, 3):
        if os.path.exists(os.path.join(path, str(i) + postfix)):
            print(os.path.join(path, str(i) + postfix))
            data.append(hs.load(os.path.join(path, str(i) + postfix)).data)
    for d in data:
        if d.shape != data[0].shape:
            return data
    else:
        data = np.stack(data, axis=0)
        data = data.swapaxes(1, 3).swapaxes(2, 4)
        return data

def _crop_from_center(data, size, com, i):
    pos_x, pos_y = com[i]

    return data[pos_y - size // 2: pos_y + size // 2,
                pos_x - size // 2: pos_x + size // 2]

def crop_from_center(data, size, com, list=False):
    if list:
        return fn_on_resized(data, _crop_from_center, size, com, list=True)
    else:
        return _crop_from_center(data, size, com, 0)
                
def _crop(data, size, position):
    return data[position[0]:position[0] + size, 
                position[1]:position[1] + size]

def crop(data, size, position):
    # this is a function that can be used to apply a crop function on a 4D array
    return fn_on_resized(data, _crop, size, position)

def rotate_by_cen(data, angle, com, i):
    # this is a function that can be used to apply a function for rotation on a 4D array
    return imutils.rotate(data, angle, center=com[i])


def fn_on_resized(data, fn, *args, **kwargs):
    # this is a function that can be used to apply a function on a 4D array
    shape = data.shape
    tmp = np.reshape(data, (-1, *shape[-2:]))
    output = []
    for i in range(tmp.shape[0]):
        if 'list' in kwargs.keys():
            output.append(fn(tmp[i], *args, i))
        else:
            output.append(fn(tmp[i], *args))
        
    output = np.array(output)
    output = np.reshape(output, (*shape[:-2], output.shape[-2], output.shape[-1]))

    return output

def get_emb(data,n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = UMAP(n_components= n_components,
                        n_neighbors = n_neighbors,
                        min_dist = min_dist, init='random')

    embedding = reducer.fit_transform(data)
    return embedding

def get_lbl(emb, n_clusters=2, gamma=0.5):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', assign_labels='kmeans', gamma=gamma)
    labels = spectral.fit_predict(emb)
    return labels

def selected_ind(data, eps=0.2, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    emb = get_emb(data.reshape((len(data), -1)))
    fit = dbscan.fit_predict(emb)
    selected_data = []

    for i in set(fit):
        selected_data.append(data[random.choice(np.where(fit == i)[0])])

    return np.array(selected_data)
