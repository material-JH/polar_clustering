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
data = load_data(r"/mnt/c/Users/em3-user/Documents/set4_Ru")
#%%
# Define the circle
size = 40  # Size of the circle
center = (size // 2, size // 2)  # Center point of the circle
x, y = np.meshgrid(np.arange(size), np.arange(size))
dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
circle = np.zeros((size, size))
circle[dist <= center[0]] = 1
#%%
com = fn_on_resized(data, get_center, circle)
com = np.reshape(com, (-1, *com.shape[-1:]))
com = [list(map(int, c[::-1])) for c in com]
    
#%%
data_post = fn_on_resized(data, rotate_by_cen, 81, com, list=True)
data_post = crop_from_center(data_post, 250, com)
#%%
com = fn_on_resized(data_post, get_center, circle)
com = np.reshape(com, (-1, *com.shape[-1:]))
com = [list(map(int, c[::-1])) for c in com]
data_post = crop_from_center(data_post, 50, com)

#%%
disk_pos_002 = [7, 100]
disk_pos_011 = [55, 145]
data_post_002 = crop(data_post, 50, disk_pos_002)
data_post_011 = crop(data_post, 50, disk_pos_011)
data_post_002_norm = normalize_Data(data_post_002)
data_post_011_norm = normalize_Data(data_post_011)

n = 7
data_post_011_norm = fn_on_resized(data_post_011_norm, cv2.GaussianBlur, (n, n), 0)
plot_vertical(data_post_002_norm)
#%%
n_neighbors = 15
n_components = 2
min_dist = 0.2

n = 5
simulations = np.load('output/disk_011_2.npy')
simulations = normalize_Data(simulations)[:,:,:,0,0]
simulations = fn_on_resized(simulations, cv2.GaussianBlur, (n, n), 0)[:,:,:,0,0]

#%%
fig, ax = plt.subplots(2, 8)
for img_gpu in range(8):
    for j in range(2):
        ax[j, img_gpu].imshow(simulations[img_gpu + j * len(simulations) // 2,:,:])
        ax[j, img_gpu].axis('off')
    
#%%
# embedding, labels = get_emb_lbl_real(data_post_002)
xyz = reduce((lambda x, y: x * y), data_post_002.shape[:3])

new = np.concatenate([data_post_011_norm.reshape(xyz , -1), simulations.reshape(len(simulations), -1)], axis=0)
#%%
# embedding, labels = get_emb_lbl(simulations.reshape(len(simulations), -1), n_neighbors=15, min_dist=0.1 * 5,)
# embedding, labels = get_emb_lbl(data_post_011_norm.reshape(xyz , -1), n_neighbors=15, min_dist=0.1, n_components=3)
embedding, labels = get_emb_lbl(new, n_neighbors=15, min_dist=0.1, n_components=2)
#%%
ax1, ax2 = 0, 1
plt.scatter(embedding[:xyz, ax1], embedding[:xyz, ax2])
plt.scatter(embedding[xyz:, ax1], embedding[xyz:, ax2])

#%%
test=np.zeros(1900)
for n in range(-len(simulations), 0):
    distances = np.linalg.norm(embedding[:xyz] - embedding[n], axis=1)
    nearest_neighbor_index = np.argmin(distances)
    print(nearest_neighbor_index)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    ax[0].imshow(data_post_011_norm.reshape(xyz , 50, 50)[nearest_neighbor_index])
    ax[0].axis('off')
    ax[1].imshow(simulations[n])
    ax[1].axis('off')
    plt.show()
    test[nearest_neighbor_index] = 1
print(np.where(test > 0.1))
test = test.reshape(data.shape[:-2])
print(np.where(test > 0.1))

#%%

#%%

labels = []
for f in os.listdir('output'):
    if f.__contains__('DP'):
        if f.__contains__('-'):
            labels.append(float(f[-6:-4]))
        else:
            labels.append(float(f[-5:-4]))

labels = np.array(labels)
labels /= max(labels)
# plt.scatter(embedding[:1900, 0], embedding[:1900, 1], c='blue', label='Cluster 1')
fig, ax = plt.subplots()
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='bwr', vmin=-1, vmax=1)
cbar = plt.colorbar(scatter)
# plt.scatter(embedding[labels > 0, 0], embedding[labels > 0, 1], alpha=labels[labels > 0], c='red', label='Cluster 2')
#%%

from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=5, affinity='rbf', assign_labels='kmeans')
labels = spectral.fit_predict(embedding)
# embedding, labels = get_emb_lbl(data_post, n_neighbors, n_components, min_dist)
alpha = 0.5
simLen = len(labels) - xyz
plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], alpha=alpha, c='purple', label='Cluster 0')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], alpha=alpha, c='#3b528b', label='Cluster 1')
plt.scatter(embedding[labels == 2, 0], embedding[labels == 2, 1], alpha=alpha, c='#20908c', label='Cluster 2')
plt.scatter(embedding[labels == 3, 0], embedding[labels == 3, 1], alpha=alpha, c='#5ac864', label='Cluster 3')
plt.scatter(embedding[labels == 4, 0], embedding[labels == 4, 1], alpha=alpha, c='yellow', label='Cluster 4')
# plt.scatter(embedding[xyz:, 0], embedding[xyz:, 1], alpha=0.8, c='Red', label='simulation')
plt.scatter(embedding[xyz:-simLen // 2, 0], embedding[xyz:-simLen // 2, 1], alpha=0.8, label='dn', c='red')
plt.scatter(embedding[xyz + simLen // 2:, 0], embedding[xyz + simLen // 2:, 1], alpha=0.8, label='up', c='blue')

# labels[labels == 3] = 2
plt.title('UMAP + K-means clustering')
plt.legend()
plt.show()
#%%
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
for n, img_gpu in enumerate(labels[:xyz].reshape(data_post_002.shape[:3])):
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
import cv2
# Convert to grayscale
# gray = cv2.cvtColor(data_post_002[0,20,0], cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise
n = 5
blur = cv2.GaussianBlur(data_post_011_norm[0,10,0] + 1, (n, n), 0)

blur = cv2.convertScaleAbs(blur)
data_post_011_norm[0,10,0][get_center(blur)] = 1e+1
plt.imshow(data_post_011_norm[0,10,0])

# print(center_of_mass_position(blur))
# edges = cv2.Canny(blur, 1, 2)
# print(center_of_mass_position(edges))
# edges[center_of_mass_position(edges)] = 1e+2
# plt.imshow(edges)
# Display the result
# %%

img = data[0,10,0]
rot = imutils.rotate(img, 81, center=get_center(img, circle)[::-1])
# %%
print(get_center(img, circle))
print(get_center(rot, circle))
# plt.imshow(rot)
# %%
plt.imshow(data_post[0,10,0])
# %%
