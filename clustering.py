#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from main import *
import cv2
# matplotlib.use('QtAgg')
#%%
data_post_011_norm = np.load('output/set2_SRO_011.npy')
data_post_011_norm = np.concatenate([data_post_011_norm, np.load('output/set4_Ru_011.npy')], axis=0)

def plot_vertical(data):
    fig, axs = plt.subplots(nrows=8, ncols=5, figsize=(8, 12))
    for i in range(0, 16, 2):
        for j in range(5):
            axs[i // 2, j].imshow(data[2, i, j])
            axs[i // 2, j].axis('off')
    plt.show()
    
plot_vertical(data_post_011_norm)
#%%

n = 11
simulations = np.load('output/disk_011_3.npy')
simulations = normalize_Data(simulations)[:,:,:,0,0]
simulations = fn_on_resized(simulations, cv2.GaussianBlur, (n, n), 0)[:,:,:,0,0]
simulations = fn_on_resized(simulations.reshape((1, 1, *simulations.shape)), resize, (50, 50))[0,0,:]
# simulations = crop(simulations.reshape((1, 1, *simulations.shape)), 42, [5, 1])[0,0,:]
#%%
def resize(img, size):
    tmp = np.zeros(img.shape)
    tmp += np.min(img)
    re = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
    tmp[:size[0], :size[1]] = re
    return tmp
# sim_resized = fn_on_resized(simulations.reshape((1, 1, *simulations.shape)), resize, (42, 38))[0,0]
sim_resized = fn_on_resized(simulations.reshape((1, 1, *simulations.shape)), resize, (50, 46))[0,0]
simulations = np.concatenate([simulations, sim_resized], axis=0)
# sim_resized = fn_on_resized(simulations.reshape((1, 1, *simulations.shape)), resize, (42, 38))[0,0]
# simulations = np.concatenate([simulations, sim_resized], axis=0)
#%%
n_row = len(simulations) // 32
fig, ax = plt.subplots(2, n_row)
for img_gpu in range(n_row):
    for j in range(2):
        ax[j, img_gpu].imshow(simulations[img_gpu + j * len(simulations) // 2,:,:])
        ax[j, img_gpu].axis('off')
    
#%%
n_neighbors = 15
n_components = 2
min_dist = 0.2
n_clusters = 8
gamma = 0.3
# embedding, labels = get_emb_lbl_real(data_post_002)
xyz = reduce((lambda x, y: x * y), data_post_011_norm.shape[:3])
new = np.concatenate([data_post_011_norm.reshape(xyz , -1), simulations.reshape(len(simulations), -1)], axis=0)
# embedding, labels = get_emb_lbl(simulations.reshape(len(simulations), -1), n_neighbors=15, min_dist=0.1 * 5,)
# embedding, labels = get_emb_lbl(data_post_011_norm.reshape(xyz , -1), n_neighbors=15, min_dist=0.1, n_components=3)
embedding = get_emb(new, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
labels = get_lbl(embedding, n_clusters=n_clusters, gamma=gamma)
#%%
ax1, ax2 = 0, 1
plt.scatter(embedding[:xyz, ax1], embedding[:xyz, ax2])
plt.scatter(embedding[xyz:, ax1], embedding[xyz:, ax2])
#%%
test=np.zeros(xyz)
for n in range(-len(simulations), 0, 100):
    distances = np.linalg.norm(embedding[:xyz] - embedding[n], axis=1)
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    fig.suptitle(f'{n} {nearest_neighbor_index}')
    ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[nearest_neighbor_index])
    ax[0].axis('off')
    ax[1].imshow(simulations[n])
    ax[1].axis('off')
    plt.show()
    test[nearest_neighbor_index] = 1

#%%
alpha = 1
alpha_sim = 0.5
simLen = len(labels) - xyz
labels_exp = labels[:xyz]
embedding_exp = embedding[:xyz]
for i in set(labels):
    plt.scatter(embedding_exp[labels_exp == i, ax1], embedding_exp[labels_exp == i, ax2], alpha=alpha, label=f'Cluster {i}')

plt.scatter(embedding[xyz:, ax1], embedding[xyz:, ax2], alpha=alpha_sim, label='simulation', c='red')

# plt.scatter(embedding_exp[labels_exp == 1, ax1], embedding_exp[labels_exp == 1, ax2], alpha=alpha, c='#424186', label='Cluster 1')
# plt.scatter(embedding_exp[labels_exp == 2, ax1], embedding_exp[labels_exp == 2, ax2], alpha=alpha, c='#2a778e', label='Cluster 2')
# plt.scatter(embedding_exp[labels_exp == 3, ax1], embedding_exp[labels_exp == 3, ax2], alpha=alpha, c='#22a785', label='Cluster 3')
# plt.scatter(embedding_exp[labels_exp == 4, ax1], embedding_exp[labels_exp == 4, ax2], alpha=alpha, c='#77d153', label='Cluster 4')
# plt.scatter(embedding_exp[labels_exp == 5, ax1], embedding_exp[labels_exp == 5, ax2], alpha=alpha, c='yellow', label='Cluster 5')
# plt.scatter(embedding[xyz:, 0], embedding[xyz:, 1], alpha=0.8, c='Red', label='simulation')
# plt.scatter(embedding[xyz + simLen // 2:, 0], embedding[xyz + simLen // 2:, 1], alpha=alpha_sim, label='up', c='blue')

# labels[labels == 3] = 2
plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()
#%%
# get nearest neighbor from simulation
wlabel = 4
rnd_num = np.random.choice(np.where(labels[:xyz] == wlabel)[0])
nearest_neighbor_index = np.argmin(np.linalg.norm(embedding[xyz:] - embedding[rnd_num], axis=1))
fig, ax = plt.subplots(1, 2, figsize=(3,2))
ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[rnd_num])
ax[0].title.set_text('real')
ax[0].axis('off')
ax[1].imshow(simulations[nearest_neighbor_index])
ax[1].title.set_text('simulation')
ax[1].axis('off')
plt.show()
#%%
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
for n, img_gpu in enumerate(labels[:xyz].reshape(data_post_011_norm.shape[:3])[5:]):
    # i = np.concatenate([i, [list([labels[1900]]) * 10]], axis=0)
    # i = np.concatenate([i, [list([labels[1901]]) * 10]], axis=0)
    im = axs[n].imshow(img_gpu)
    axs[n].axis('off')
from matplotlib.ticker import MaxNLocator
cbar1 = fig.colorbar(im, ax=axs[n], format='%d')    
cbar1.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
#%%
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
for img_gpu, j in zip(lbl_reshape, data.sum(axis=(-1, -2))):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axs[0].imshow(img_gpu)
    axs[1].imshow(j)
    plt.show()
# %%
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig, ax = plt.subplots(1, 2, figsize=(3, 3), dpi=100)
t = np.arange(0, 3, .01)

position = np.zeros(data.shape[1:3])
ax[0].imshow(position)
size = 150
ax[1].imshow(imutils.rotate(data[2, 0, 0], int(82), com)[com[0] - size:com[0] + size, com[1] - size:com[1] + size])
ax[0].axis('off')
ax[1].axis('off')

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)

com = get_center(data[2, 10, 0], circle)
com = list(map(int, com))
def update_frequency(new_val):
    new_val = int(new_val)
    position = np.zeros(data.shape[1:3])
    position[new_val % 38, new_val // 38] = 1
    ax[0].clear()
    ax[1].clear()
    # ax.imshow(imutils.rotate(data[0, 20, 0], int(new_val), com[200]))
    # ax.imshow(imutils.rotate(data.sum(axis=2)[2, new_val], int(82), com[200]))
    ax[0].imshow(position)
    ax[1].imshow(imutils.rotate(data[2, new_val % 38, new_val // 38], int(82), com)[com[0] - size:com[0] + size, com[1] - size:com[1] + size])
    ax[0].axis('off')
    ax[1].axis('off')
    # required to update canvas and attached toolbar!
    canvas.draw()

slider_update = tkinter.Scale(root, from_=0, to=90, orient=tkinter.HORIZONTAL,
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