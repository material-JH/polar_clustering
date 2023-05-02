#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi

#%%
data_post_exp = np.load('output/set1_SRO_002.npy')
# data_post_exp = np.load('output/Fe2O3.npy')

data_post_exp2 = np.load('output/set4_Ru_002.npy')
data_post_exp = np.concatenate((data_post_exp, data_post_exp2), axis=0)

#%%
data_post_exp = fn_on_resized(data_post_exp, normalize_Data)
# data_post_exp2 = fn_on_resized(data_post_exp2, normalize_Data)
# %%
imstack_train = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])
# imstack_train2 = data_post_exp2.reshape(-1, 50, 50)
#%%
plot_tk(imstack_train)
#%%
import cv2
simulations_sep = np.load('output/disk_002_dft.npz')
simulations = {}
for k, v in simulations_sep.items():
    simulations[k] = v

n = 5
for k, v in simulations.items():
    # simulations[k] = fn_on_resized(v, cv2.GaussianBlur, (n, n), 0)
    simulations[k] = normalize_Data(simulations[k])

simulations_tot = np.stack(list(simulations.values()), axis=0)
#%%
blurred = cv2.GaussianBlur(simulations_tot, (11, 11), 0)

simulations_blurred = np.concatenate([simulations_tot, blurred], axis=0)

#%%
# Intitialize rVAE model
input_dim = imstack_train.shape[1:]
rvae = aoi.models.rVAE(input_dim, latent_dim=2,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,)

if os.path.exists('output/rvae_002_norm.tar'):
    rvae.load_weights('output/rvae_002_norm.tar')
    print('loaded weights')

#%%
rvae.fit(
    np.concatenate((imstack_train, simulations_tot), axis=0),
    training_cycles=50,
    batch_size=2 ** 8)

#%%
rvae.fit(
    imstack_train, 
    training_cycles=100,
    batch_size=2 ** 8)

#%%
rvae.save_weights('output/rvae_002_norm')
# rvae.save_weights('output/rvae_fe2o3_norm')
#%%
rvae.manifold2d(cmap='viridis', figsize=(10, 10))
#%%
encoded_mean, encoded_sd  = rvae.encode(imstack_train)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

# encoded_mean, encoded_sd  = rvae.encode(imstack_train2)
# z21, z22, z23 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]
#%%
plot_tk(simulations_tot)
# %%
sim_mean, sim_sd  = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]


#%%
plt.scatter(z11, z13[:,1], alpha=.1, color='red')
plt.scatter(z31, z33[:,1], alpha=.1, color='green')


# %%
xyz = reduce((lambda x, y: x * y), data_post_exp.shape[:3])

for n in range(xyz):
    distances = np.linalg.norm(z33 - z13[n], axis=1)
    if np.min(distances) > 0.01:
        continue
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('exp vs sim')
    ax[0].imshow(data_post_exp.reshape(xyz, 50, 50)[n])
    ax[0].axis('off')
    # ax[1].imshow(simulations_tot[near[nearest_neighbor_index]])
    ax[1].imshow(simulations_tot[nearest_neighbor_index])
    ax[1].axis('off')
    plt.show()
# %%
xyz = reduce((lambda x, y: x * y), data_post_exp.shape[:3])

is_closed = []
for n in range(xyz):
    distances = np.linalg.norm(z33 - z13[n], axis=1)
    # distances = np.linalg.norm(z33 - z13[n], axis=1) + abs(z31 - z11[n]) / 10
    if np.min(distances) > .01:
        is_closed.append(0)
    else:
        is_closed.append(1)

is_closed = np.array(is_closed)

fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(8, 4))
for n, img in enumerate(is_closed.reshape(data_post_exp.shape[:3])):
    im = axs[n].imshow(img)
    axs[n].axis('off')
plt.show()
#%%
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(12, 6))
for n, img in enumerate(z13.reshape((*data_post_exp.shape[:3], 2))):
    img = img - img.min()
    img = img / img.max()
    img = 1 - img
    # img = np.stack([img[:,:,0], np.ones(img.shape[:2]), img[:,:,1]])
    img = np.concatenate([img, np.ones((*img.shape[:2], 1))], axis=-1)
    # img = np.moveaxis(img, 0, -1)
    axs[n].imshow(img)
    # im1 = axs[n].imshow(img[:,:,0], cmap='Reds', origin='lower', alpha=0.5, label='Channel 1',vmin=-1, vmax=1)
    # im2 = axs[n].imshow(img[:,:,1], cmap='Blues', origin='lower', alpha=0.5, label='Channel 2',vmin=-1, vmax=1)
    # im3 = axs[n].imshow(im[1], cmap='Greens', origin='lower', alpha=0.3, label='Channel 2',vmin=-1, vmax=4)
    axs[n].axis('off')
plt.show()

# %%
tot = np.concatenate((imstack_train, simulations_tot), axis=0)
means = []
stds = []
for i in range(tot.shape[0]):
    means.append(tot[i].mean())
    stds.append(tot[i].std())
fig, ax = plt.subplots()
ax.errorbar(range(tot.shape[0]), means, yerr=stds, fmt='o', capsize=5)

# %%

img = z13.reshape((*data_post_exp.shape[:2], 3))
print(img.shape)

#%%
im1 = plt.imshow(img[:,:,0], cmap='Reds', origin='lower', alpha=0.3, label='Channel 1',vmin=-1.5, vmax=1)
im2 = plt.imshow(img[:,:,1], cmap='Blues', origin='lower', alpha=0.3, label='Channel 2',vmin=-1.5, vmax=1)
im3 = plt.imshow(img[:,:,2], cmap='Greens', origin='lower', alpha=0.3, label='Channel 3',vmin=-1.5, vmax=1)
plt.ylim(img.shape[0], 0)
plt.show()
# %%


import matplotlib.pyplot as plt
import numpy as np

# create a 3x3 RGB image with values between -1 and 2
image_data = np.random.uniform(low=-1, high=2, size=(3, 3, 3))

# normalize the data to the range [0, 1]
image_data = (image_data - (-1)) / (2 - (-1))

# create the plot
plt.imshow(image_data)
plt.show()
# %%
im = im - im.min()
#%%
im = im / im.max()
# %%
im = np.swapaxes(im, 0, 2)
# %%
