#%%
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from sklearn.cluster import DBSCAN
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

a = data_post_011_norm[0,25,0]
b = simulations[1350]
fig, ax = plt.subplots(2,2)

ax[0,0].imshow(a)
ax[1,0].imshow(b)
ax[0,1].hist(a.flatten(), bins = 100)
ax[1,1].hist(b.flatten(), bins = 100)

plt.show()
#%%
eps = 0.2
simulations_dn = np.load('output/disk_011_4_dn.npy')
simulations_dn = select_data(simulations_dn, eps=eps, min_samples=5)
simulations = crop(simulations_dn, 50, [0, 0])
for i in range(0, 5, 2):
    for j in range(0, 5, 2):
        if i == 0 and j == 0:
            continue
        simulations = np.concatenate([simulations, crop(simulations_dn, 50, [i, j])], axis=0)
num_dn = len(simulations)
simulations_up = np.load('output/disk_011_4_up.npy')
simulations_up = select_data(simulations_up, eps=eps, min_samples=5)

for i in range(0, 5, 1):
    for j in range(0, 5, 1):
        simulations = np.concatenate([simulations, crop(simulations_up, 50, [i, j])], axis=0)

print(num_dn, len(simulations) - num_dn)
#%%
n =15
simulations = fn_on_resized(simulations, cv2.GaussianBlur, (n, n), 0)
simulations = normalize_Data(simulations)
simulations = fn_on_resized(simulations, cv2.GaussianBlur, (n, n), 0)
simulations = fn_on_resized(simulations, resize, (50, 50))
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
sim_resized = fn_on_resized(simulations.reshape((1, 1, *simulations.shape)), resize, (50, 48))[0,0]
simulations = np.concatenate([simulations, sim_resized], axis=0)
#%%
n_neighbors = 15
n_components = 2
min_dist = 0.1
n_clusters = 8
gamma = 0.3
# embedding, labels = get_emb_lbl_real(data_post_002)
xyz = reduce((lambda x, y: x * y), data_post_011_norm.shape[:3])
new = np.concatenate([data_post_011_norm.reshape(xyz , -1), simulations.reshape(len(simulations), -1)], axis=0)
new = new ** 3
# embedding, labels = get_emb_lbl(simulations.reshape(len(simulations), -1), n_neighbors=15, min_dist=0.1 * 5,)
# embedding, labels = get_emb_lbl(data_post_011_norm.reshape(xyz , -1), n_neighbors=15, min_dist=0.1, n_components=3)
embedding = get_emb(new, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
labels = get_lbl(embedding, n_clusters=n_clusters, gamma=gamma)
#%%
ax1, ax2 = 0, 1
plt.scatter(embedding[:xyz // 2, ax1], embedding[:xyz // 2, ax2], label='SRO')
plt.scatter(embedding[xyz // 2:xyz, ax1], embedding[xyz // 2:xyz, ax2], label='Ru')
plt.scatter(embedding[xyz:, ax1], embedding[xyz:, ax2], label='sim')
plt.legend()
#%%
test=np.zeros(xyz)
for n in range(-len(simulations), 0):
    distances = np.linalg.norm(embedding[:xyz] - embedding[n], axis=1)
    if np.min(distances) > 0.05:
        continue
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('exp vs sim')
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

plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()
#%%
# get nearest neighbor from simulation
wlabel = 7
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
from matplotlib.ticker import MaxNLocator

for m in range(2):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
    for n, img in enumerate(labels[:xyz].reshape(data_post_011_norm.shape[:3])[range(m * 5, m * 5 + 5)]):
        # i = np.concatenate([i, [list([labels[1900]]) * 10]], axis=0)
        # i = np.concatenate([i, [list([labels[1901]]) * 10]], axis=0)
        im = axs[n].imshow(img, vmin=0, vmax=n_clusters)
        axs[n].axis('off')
    cbar1 = fig.colorbar(im, ax=axs[n], format='%d')    
    cbar1.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

#%%
ind_closed = []
for emb in embedding[:xyz]:
    distances = np.linalg.norm(embedding[xyz:] - emb, axis=1)
    if np.min(distances) > 0.2:
        ind_closed.extend(np.where(distances == np.min(distances))[0])

ind_closed = np.unique(ind_closed)
simulations = simulations[ind_closed]
print(ind_closed)
#%%

is_closed = []
for emb in embedding[:xyz]:
    distances = np.linalg.norm(embedding[xyz:] - emb, axis=1)
    if np.min(distances) > 0.2:
        is_closed.append(0)
    else:
        is_closed.append(1)

is_closed = np.array(is_closed)

for m in range(2):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
    for n, img in enumerate(is_closed.reshape(data_post_011_norm.shape[:3])[range(m * 5, m * 5 + 5)]):
        im = axs[n].imshow(img)
        axs[n].axis('off')
    plt.show()

#%%
alpha = 0.1
alpha_sim = 0.5
labels_exp = labels[:xyz]
embedding_exp = embedding[:xyz]
plt.scatter(embedding_exp[:xyz // 2,0], embedding_exp[:xyz //2, 1], alpha=alpha, label=f'set2')
plt.scatter(embedding_exp[xyz // 2:,0], embedding_exp[xyz //2:, 1], alpha=alpha, label=f'set4')

plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()

#%%