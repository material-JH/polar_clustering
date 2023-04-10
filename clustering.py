#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from main import *
import cv2
#%%
data_post_011_norm = np.load('output/set2_SRO_011.npy')
data_post_011_norm = np.concatenate([data_post_011_norm, np.load('output/set4_Ru_011.npy')], axis=0)

#%%
plot_vertical(data_post_011_norm[5])
#%%
eps = 0.2
import pickle

with open('output/sep_011_5.pkl', 'rb') as f:
    sep = pickle.load(f)

simulations_sep = np.load('output/disk_011_5.npz')
concatenated_array = np.concatenate(list(simulations_sep.values()), axis=0)

simulations = {}

#%%

for k, v in simulations_sep.items():
    simulations[k] = v[::2]

for k, v in simulations.items():
    tmp = crop(v[::50], 50, [0, 0])
    for i in range(0, 7, 3):
        for j in range(0, 7, 3):
            if i == 0 and j == 0:
                continue
            tmp = np.concatenate([tmp, crop(v, 50, [i, j])], axis=0)
    simulations[k] = tmp


#%%
n = 11
for k, v in simulations.items():
    # simulations[k] = fn_on_resized(v, cv2.GaussianBlur, (n, n), 0)
    simulations[k] = normalize_Data(simulations[k])

simulations_tot = np.concatenate(list(simulations.values()), axis=0)
#%%

simulations_rnd = simulations_tot[np.random.choice(len(simulations_tot), size=75)]
plot_vertical(simulations_rnd.reshape(len(simulations_rnd) // 5, 5, 50, 50))

#%%
n_neighbors = 15
n_components = 2
min_dist = 0.5
n_clusters = 8
gamma = 0.3
# embedding, labels = get_emb_lbl_real(data_post_002)
xyz = reduce((lambda x, y: x * y), data_post_011_norm.shape[:3])
new = data_post_011_norm.reshape(xyz , -1)
new = new.astype(np.float16)
for k, v in simulations.items():
    new = np.concatenate([new, v.reshape(len(v), -1)], axis=0)
# new = new ** 2
# embedding, labels = get_emb_lbl(simulations.reshape(len(simulations), -1), n_neighbors=15, min_dist=0.1 * 5,)
# embedding, labels = get_emb_lbl(data_post_011_norm.reshape(xyz , -1), n_neighbors=15, min_dist=0.1, n_components=3)
embedding = get_emb(new, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
labels = get_lbl(embedding, n_clusters=n_clusters, gamma=gamma)
# from cuml.cluster import DBSCAN as cuDBSCAN
# dbscan_float = cuDBSCAN(eps=1.0, min_samples=30)
# dbscan_float.fit(embedding)
# labels = dbscan_float.labels_

print(len(set(labels)))
#%%
emb_exp = embedding[:xyz]
emb_sim = {}

for k, v in simulations.items():
    emb_sim[k] = embedding[xyz + len(v) * sep[k]: xyz + len(v) * (sep[k] + 1)]

#%%
ax1, ax2 = 0, 1
plt.scatter(embedding[:xyz // 2, ax1], embedding[:xyz // 2, ax2], label='SRO')
plt.scatter(embedding[xyz // 2:xyz, ax1], embedding[xyz // 2:xyz, ax2], label='Ru')
for k, v in emb_sim.items():
    plt.scatter(v[:, ax1], v[:, ax2], label=f'sim_{k}', alpha=0.01, c='red')
plt.legend()
#%%
test=np.zeros(xyz)
for n in range(-len(simulations_tot), 0):
    distances = np.linalg.norm(embedding[:xyz] - embedding[n], axis=1)
    if np.min(distances) > 0.1:
        continue
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('exp vs sim')
    ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[nearest_neighbor_index])
    ax[0].axis('off')
    ax[1].imshow(simulations_tot[n])
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

simulations_tot = np.concatenate(list(simulations.values()), axis=0)
wlabel = 0
# rnd_num = np.random.choice(np.where(labels[:xyz] == wlabel)[0])
#%%
rnd_num = np.random.choice(range(len(emb_exp)))
nearest_neighbor_index = np.argmin(np.linalg.norm(embedding[xyz:] - embedding[rnd_num], axis=1))
distances = np.linalg.norm(embedding[xyz:] - embedding[rnd_num], axis=1)
while np.min(distances) > 0.2:
    rnd_num = np.random.choice(range(len(emb_exp)))
    nearest_neighbor_index = np.argmin(np.linalg.norm(embedding[xyz:] - embedding[rnd_num], axis=1))
    distances = np.linalg.norm(embedding[xyz:] - embedding[rnd_num], axis=1)

print(np.min(distances), rnd_num, nearest_neighbor_index)
fig, ax = plt.subplots(1, 2, figsize=(3,2))
ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[rnd_num])
ax[0].title.set_text('real')
ax[0].axis('off')
ax[1].imshow(simulations_tot[nearest_neighbor_index])
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