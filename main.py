import random
import imutils
import hyperspy.api as hs
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from functools import reduce

try:
    from cuml.manifold import UMAP
    from cuml.cluster import DBSCAN 
    import cupyx.scipy.signal as sig
    import cupy as cp
except:
    from sklearn.manifold import UMAP
    from sklearn.cluster import DBSCAN
    import scipy.signal as sig

def get_circle_conv(size):
    center = (size // 2, size // 2)  # Center point of the circle
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    circle = np.zeros((size, size))
    circle[dist <= center[0]] = 1
    return circle

def plot_vertical(data):
    shape = data.shape
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(8, 12))
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

def select_data(data, eps=0.2, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    emb = get_emb(data.reshape((len(data), -1)))
    fit = dbscan.fit_predict(emb)
    selected_data = []

    for i in set(fit):
        selected_data.append(data[random.choice(np.where(fit == i)[0])])

    return np.array(selected_data)

def plot_tk(data):

    import tkinter

    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    # Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.figure import Figure

    import numpy as np
    import cv2

    root = tkinter.Tk()
    root.wm_title("Embedding in Tk")

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)

    ax.imshow(data[0])

    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()

    canvas.mpl_connect(
        "key_press_event", lambda event: print(f"you pressed {event.key}"))
    canvas.mpl_connect("key_press_event", key_press_handler)

    button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)

    def update_frequency(new_val):
        new_val = int(new_val)
        ax.clear()
        # ax.imshow(imutils.rotate(data[0, 20, 0], int(new_val), com[200]))
        # ax.imshow(imutils.rotate(data.sum(axis=2)[2, new_val], int(82), com[200]))
        ax.imshow(data[new_val])
        ax.axis('off')
        # required to update canvas and attached toolbar!
        canvas.draw()

    slider_update = tkinter.Scale(root, from_=0, to=len(data), orient=tkinter.HORIZONTAL,
                                command=update_frequency, label="Frequency [Hz]")

    # Packing order is important. Widgets are processed sequentially and if there
    # is no space left, because the window is too small, they are not displayed.
    # The canvas is rather flexible in its size, so we pack it last which makes
    # sure the UI controls are displayed as long as possible.
    button_quit.pack(side=tkinter.BOTTOM)
    slider_update.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
    toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    tkinter.mainloop()

def plot_random(data):
    fig, ax = plt.subplots(3, 3, figsize=(3, 3), dpi=100)
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(data[random.randint(0, len(data))])
            ax[i, j].axis('off')
    plt.show()