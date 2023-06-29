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
filename = 'weights/conv_mmdpvae_1.3_norm'
import gc
gc.collect()
rvae = MMD_regpVAE(input_dim, latent_dim=10, h_dim=128,
                        p_weight=1e+3, m_weight=1e+2,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.2, div_weight=.2, cont_weight=20, filename=filename)
encoder = EncoderNet(1, hidden_channel=128, latent_dim=10)
rvae.set_encoder(encoder)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%
ind_test = np.random.choice(range(len(imstack_train)), len(imstack_train) // 5, replace=False)
ind_train = np.setdiff1d(range(len(imstack_train)), ind_test)

#%%
rvae.fit(
    X_train= imstack_train[ind_train],
    y_train= imstack_polar[ind_train],
    X_test = imstack_train[ind_test],
    y_test = imstack_polar[ind_test],
    training_cycles=100,
    batch_size=2 ** 7,
    filename=filename)

# %%

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
plt.hist2d(polarization_keys_sim, p, bins=100)
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
for i in range(6):
    if i == 5:
        plt.colorbar(pcm, ax=ax[i], aspect=30)
        ax[i].axis('off')
    else:
        pcm = ax[i].imshow(imgs[i, ...], cmap='RdBu_r', vmin=pmin, vmax=pmax, aspect='auto')
        ax[i].axis('off')
# %%
imgs = data_post_exp.sum(axis=(-1, -2))
pmin = 0
pmax = imgs.max()
fig, ax = plt.subplots(1, 6)
for i in range(6):
    if i == 5:
        plt.colorbar(pcm, ax=ax[i], aspect=30)
        ax[i].axis('off')
    else:
        pcm = ax[i].imshow(imgs[i, ...], cmap='RdBu_r', vmin=pmin, vmax=pmax, aspect='auto')
        # ax[i].axis('off')

# %%
