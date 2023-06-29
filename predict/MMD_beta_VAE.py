#%%
import os
os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from model.vae_model import *
from lib.plot import *
#%%
data_post_exp = np.load('output/set1_Ru_002.npy')
data_post_exp4 = np.load('output/set4_SRO_002.npy')
#%%
data_stack = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])
data_stack4 = data_post_exp4.reshape(-1, data_post_exp4.shape[-2], data_post_exp4.shape[-1])
#%%

polarization_keys_exp = [torch.inf for _ in range(len(data_stack))]
polarization_keys_exp4 = [torch.inf for _ in range(len(data_stack4))]

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_tot = np.stack(simulations_sep.values(), axis=0)
#%%
polarization_keys_sim = [float(k.split('_')[2]) for k in simulations_sep.keys()]
polarization_keys_sim = np.array(polarization_keys_sim)
p_mean = np.mean(polarization_keys_sim)
p_std = np.std(polarization_keys_sim)
polarization_keys_sim = (polarization_keys_sim - p_mean) / p_std
#%%
# Intitialize rVAE model
from imutils import resize
data_stack = fn_on_resized(data_stack, resize, 64, 64)
data_stack4 = fn_on_resized(data_stack4, resize, 64, 64)
simulations_tot = fn_on_resized(simulations_tot, resize, 64, 64)
#%%
data_stack = fn_on_resized(data_stack, normalize_Data)
data_stack4 = fn_on_resized(data_stack4, normalize_Data)
simulations_tot = fn_on_resized(simulations_tot, normalize_Data)

imstack_train = np.concatenate((data_stack, simulations_tot), axis=0)
imstack_train = np.concatenate((imstack_train, data_stack4), axis=0)
# imstack_train = np.concatenate((imstack_train, simulations_tot), axis=0)
imstack_polar = np.concatenate((polarization_keys_exp, polarization_keys_sim), axis=0)
imstack_polar = np.concatenate((imstack_polar, polarization_keys_exp4), axis=0)
# imstack_polar = np.concatenate((imstack_polar, polarization_keys_sim), axis=0)
input_dim = imstack_train.shape[-2:]
#%%
filename = 'weights/conv_mmdpvae_1.1_norm'
import gc
gc.collect()
rvae = MMD_regpVAE(input_dim, latent_dim=10, h_dim=128,
                        p_weight=5e+2, m_weight=5e+2,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
encoder = EncoderNet(1, hidden_channel=128, latent_dim=10)
rvae.set_encoder(encoder)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%
z_sim, _ = rvae.encode(simulations_tot)
z, _ = rvae.encode(data_stack)
z4, _ = rvae.encode(data_stack4)
#%%
decoded_sim = rvae.decode(z_sim)
decoded = rvae.decode(z)
decoded4 = rvae.decode(z4)
#%%
tot = 0
for n in range(100, 1900):
    distances = np.linalg.norm(z_sim - z[n], axis=1)
    # if np.min(distances) > 0.2:
    #     continue
    tot += 1
    if tot > 10:
        break
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(2, 2, figsize=(5,5))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    ax[0,0].imshow(data_stack[n])
    ax[0,0].axis('off')
    ax[1,0].imshow(simulations_tot[nearest_neighbor_index])
    ax[1,0].axis('off')
    ax[0,1].imshow(decoded[n])
    ax[0,1].axis('off')
    ax[1,1].imshow(decoded_sim[nearest_neighbor_index])
    ax[1,1].axis('off')
#%%
fig, ax = plt.subplots(1, 3)
ind = 1500
ax[0].imshow(decoded_sim[ind])
ax[1].imshow(decoded[ind])
ax[2].imshow(decoded4[ind])

# %%
from cuml.manifold import UMAP
reducer = UMAP()
embedding = reducer.fit_transform(np.concatenate((z, z_sim, z4), axis=0))
#%%
plt.scatter(embedding[:len(z), 0], embedding[:len(z), 1], label='exp', alpha=.1, s=5)
plt.scatter(embedding[len(z):len(z)+len(z_sim), 0], embedding[len(z):len(z)+len(z_sim), 1], label='sim', alpha=.1, s=5)
plt.scatter(embedding[len(z)+len(z_sim):, 0], embedding[len(z)+len(z_sim):, 1], label='exp4', alpha=.1, s=5)
# %%
p = rvae.fcl_net(torch.tensor(z).to(rvae.device)).detach().cpu().numpy()
p4 = rvae.fcl_net(torch.tensor(z4).to(rvae.device)).detach().cpu().numpy()
p_sim = rvae.fcl_net(torch.tensor(z_sim).to(rvae.device)).detach().cpu().numpy()
# %%

plt.hist2d(polarization_keys_sim, p_sim, bins=100)
# %%
p_plot = p * p_std + p_mean
imgs = p_plot.reshape(5, 38, 10)
pmin = -0.03
pmax = 0.03
fig, ax = plt.subplots(1, 6)
for n in range(6):
    if n == 5:
        plt.colorbar(pcm, ax=ax[n], aspect=30)
        # ax[i].axis('off')
    else:
        pcm = ax[n].contourf(imgs[n, ...], cmap='RdBu', vmin=pmin, vmax=pmax, levels=50)
        ax[n].set_ylim(38, 0)
        # ax[i].axis('off')
# %%
data_post_exp = np.load('output/set1_Ru_002.npy')
data_post_expm = np.load('output/set1_Ru_m002.npy')
imgs = data_post_exp.sum(axis=(-1, -2))
imgs2 = data_post_expm.sum(axis=(-1, -2))
vmin = min(imgs.min(), imgs2.min())
vmax = max(imgs.max(), imgs2.max())

fig, ax = plt.subplots(1,4, figsize=(6, 5))
ylim = np.array([34, 0])
for n, img in enumerate(imgs2):
    if n % 2 == 1:
        continue
    n = n // 2
    ax[n].contourf(img, alpha=0.6, cmap='Reds', vmin=vmin, vmax=vmax, levels=50)
    ax[n].set_ylim(*ylim)
    if n == 0:
        ylim += 2
        ax[n].set_ylim(*ylim)
        ylim -= 2
    if n == 1:
        ylim += 1
        ax[n].set_ylim(*ylim)
        ylim -= 1
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    x = range(38)
    if n == 0:
        x = range(0 - 2, 38 - 2)
        c = 'b'
    elif n == 1:
        x = range(0 - 1, 38 - 1)
        c = 'gray'
    elif n == 2:
        c = 'r'
    ax[-1].plot(np.mean(img, axis=1), x, c=c, alpha=.5)
    ax[-1].set_xlim(vmin, vmax)
    ax[-1].set_ylim(*ylim)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
# %%
