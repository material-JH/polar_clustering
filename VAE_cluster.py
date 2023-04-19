#%%
import numpy as np
import matplotlib.pyplot as plt
from main import *
#%%
data_post_exp = np.load('output/set2_SRO_002.npy')
data_post_exp2 = np.load('output/set4_Ru_011.npy')
data_post_exp = np.concatenate((data_post_exp, data_post_exp2), axis=0)

# %%
import atomai as aoi
imstack_train = data_post_exp.reshape(-1, 50, 50)
imstack_train2 = data_post_exp2.reshape(-1, 50, 50)
#%%
# Intitialize rVAE model
input_dim = (50, 50)
rvae = aoi.models.rVAE(input_dim, latent_dim=2,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,)

if os.path.exists('output/rvae_002.tar'):
    rvae.load_weights('output/rvae_002.tar')
    print('loaded weights')
# Train

#%%
rvae.fit(
    imstack_train, 
    training_cycles=200,
    batch_size=2 ** 8)
#%%
rvae.save_weights('output/rvae_002')
#%%
rvae.manifold2d(cmap='viridis', figsize=(10, 10))
#%%
encoded_mean, encoded_sd  = rvae.encode(imstack_train)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

encoded_mean, encoded_sd  = rvae.encode(imstack_train2)
z21, z22, z23 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]
# %%
import cv2
simulations_sep = np.load('output/disk_002_dps.npz')
simulations = {}
for k, v in simulations_sep.items():
    simulations[k] = v

n = 11
for k, v in simulations.items():
    simulations[k] = fn_on_resized(v, cv2.GaussianBlur, (n, n), 0)
    simulations[k] = normalize_Data(simulations[k])

simulations_tot = np.concatenate(list(simulations.values()), axis=0)
print(simulations_tot.shape)
# %%
sim_mean, sim_sd  = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]


# %%
plt.scatter(z13[:,0], z13[:,1], alpha=.1, color='red', label='exp')
plt.scatter(z23[:,0], z23[:,1], alpha=.1, color='blue', label='exp2')
c = ['orange', 'purple', 'cyan']
for c, m in zip(c, simulations.items()):
    k, v = m
    sim_mean, sim_sd  = rvae.encode(v)
    z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]
    plt.scatter(z33[:,0], z33[:,1], alpha=.1, label=k, color=c)
# plt.scatter(z33[:,0], z33[:,1], alpha=.1, color='green', label='sim')
plt.legend()
#%%
plt.scatter(z11, z13[:,1], alpha=.1, color='red')
plt.scatter(z21, z23[:,1], alpha=.1, color='blue')
plt.scatter(z31, z33[:,1], alpha=.1, color='green')


# %%
xyz = reduce((lambda x, y: x * y), data_post_exp2.shape[:3])

for n in range(xyz):
    distances = np.linalg.norm(z33 - z23[n], axis=1) + abs(z31 - z21[n])
    if np.min(distances) > 0.02:
        continue
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(1, 2, figsize=(3,2))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('exp vs sim')
    ax[0].imshow(data_post_exp2.reshape(xyz , 50, 50)[n])
    ax[0].axis('off')
    # ax[1].imshow(simulations_tot[near[nearest_neighbor_index]])
    ax[1].imshow(simulations_tot[nearest_neighbor_index])
    ax[1].axis('off')
    plt.show()
# %%
xyz = reduce((lambda x, y: x * y), data_post_exp.shape[:3])

is_closed = []
for n in range(xyz):
    # distances = np.linalg.norm(z33 - z13[n], axis=1) + abs(z31 - z21[n])
    distances = np.linalg.norm(z33 - z13[n], axis=1)
    if np.min(distances) > .1:
        is_closed.append(0)
    else:
        is_closed.append(1)

is_closed = np.array(is_closed)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
for n, img in enumerate(is_closed.reshape(data_post_exp.shape[:3])):
    im = axs[n].imshow(img)
    axs[n].axis('off')
plt.show()
#%%
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(8, 4))
for n, img in enumerate(z13.reshape((*data_post_exp.shape[:3], 2))):
    im = np.swapaxes(img, 0, 2)
    im = np.swapaxes(im, 1, 2)

    im1 = axs[n].imshow(im[0], cmap='Reds', origin='lower', alpha=0.7, label='Channel 1')
    im2 = axs[n].imshow(im[1], cmap='Blues', origin='lower', alpha=0.7, label='Channel 2')

    axs[n].axis('off')
plt.show()
