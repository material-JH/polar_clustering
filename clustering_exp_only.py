#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from main import *
import cv2
#%%
data_post_011_norm = np.load('output/set2_SRO_011.npy')
data_post_011_norm = np.concatenate([data_post_011_norm, np.load('output/set4_Ru_011.npy')], axis=0)

def fft2d(data):
    return np.fft.fftshift(np.fft.fft2(data))

def ifft2d(data):
    return normalize_Data(np.fft.ifft2(data))

data_fft = fn_on_resized(data_post_011_norm, fft2d)
data_ifft = fn_on_resized(data_fft, ifft2d)

#%%
plot_tk(np.reshape(np.abs(data_ifft), (-1, 50, 50)))
#%%
plt.imshow(np.abs(data_ifft[0,15,0]), vmin=0, vmax=3)
plt.show()
plt.imshow(data_post_011_norm[0,15,0], vmin=0, vmax=3)


#%%
n_neighbors = 30
n_components = 2
min_dist = 0.01
n_clusters = 6
gamma = 0.1
# embedding, labels = get_emb_lbl_real(data_post_002)
xyz = reduce((lambda x, y: x * y), data_ifft.shape[:3])
new = data_ifft.reshape(xyz , -1)
new = new.astype(np.float16)
embedding = get_emb(new, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
labels = get_lbl(embedding, n_clusters=n_clusters, gamma=gamma)
#%%
ax1, ax2 = 0, 1
plt.scatter(embedding[:xyz // 2, ax1], embedding[:xyz // 2, ax2], label='SRO', alpha=0.5)
plt.scatter(embedding[xyz // 2:xyz, ax1], embedding[xyz // 2:xyz, ax2], label='Ru', alpha=0.5)
plt.legend()
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
test=np.zeros(xyz)
for n in range(-xyz // 2, 0):
    distances = np.linalg.norm(embedding[:xyz // 2, 1:] - embedding[n, 1:], axis=1)
    if np.min(distances) > 0.01:
        continue
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('exp vs sim')
    ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[nearest_neighbor_index])
    ax[0].axis('off')
    ax[1].imshow(data_post_011_norm.reshape(xyz , 50, 50)[n])
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
simulations_len = [len(v) for v in simulations.values()]
is_closed = []
for emb in embedding[:xyz]:
    distances = np.linalg.norm(embedding[xyz:] - emb, axis=1)
    if np.min(distances) > 0.2:
        is_closed.append(0)
    else:
        min_d = np.argmin(distances)
        for i, v in enumerate(simulations_len):
            if min_d < v:
                is_closed.append(i + 1)
                break
            else:
                min_d -= v

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