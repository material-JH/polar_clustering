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
data_post_exp = np.stack([data_post_exp, 
                        fn_on_resized(np.load('output/set1_Ru_m002.npy'), np.flip, 0)], axis=-3)
data_post_exp4 = np.load('output/set4_SRO_002.npy')
data_post_exp4 = np.stack([data_post_exp4, 
                        fn_on_resized(np.load('output/set4_SRO_m002.npy'), np.flip, 0)], axis=-3)
#%%
data_stack = data_post_exp.reshape(-1, 2, data_post_exp.shape[-2], data_post_exp.shape[-1])
data_stack4 = data_post_exp4.reshape(-1, 2, data_post_exp4.shape[-2], data_post_exp4.shape[-1])
#%%

polarization_keys_exp = [torch.inf for _ in range(len(data_stack))]
polarization_keys_exp4 = [torch.inf for _ in range(len(data_stack4))]

#%%
simulations_positive = np.stack(np.load('output/disk_002_dft.npz').values(), axis=0)
simulations_negative = np.stack(np.load('output/disk_m002_dft.npz').values(), axis=0)
simulations_tot = np.stack([simulations_positive,
                            fn_on_resized(simulations_negative, np.flip, 0)], axis=-3)
#%%
polarization_keys_sim = [float(k.split('_')[2]) for k in np.load('output/disk_002_dft.npz').keys()]
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
from copy import deepcopy
filename = 'weights/conv_mmdpvae2_1.0_norm'
import gc
gc.collect()
rvae = MMD_regpVAE2(input_dim, latent_dim=10, h_dim=128,
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

#%%
def pred_p(self, x: torch.Tensor | np.ndarray):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    z_mean_all = torch.zeros((x.shape[0], self.z_dim * 2)).to(self.device)
    for i in range(2):
        xs = x[:,i,:,:]
        with torch.no_grad():
            z_mean, z_logsd = self.encode(xs)
            z_mean = torch.tensor(z_mean, dtype=torch.float32)
            z_mean_all[:, i * (self.z_dim): (i + 1) * (self.z_dim)] = z_mean
    with torch.no_grad():
        p = self.fcl_net(z_mean_all)
    return p.cpu().numpy()

# %%
p = pred_p(rvae, simulations_tot)
# %%
plt.hist2d(p, polarization_keys_sim, bins=50)
# %%
p = pred_p(rvae, data_stack)
p4 = pred_p(rvae, data_stack4)
# %%

p_plot = p4 * p_std + p_mean
imgs = p_plot.reshape(5, 38, 10)
vmin = -0.03
vmax = 0.03
fig, ax = plt.subplots(1, 6)
for i in range(6):
    if i == 5:
        plt.colorbar(pcm, ax=ax[i], aspect=30)
        ax[i].axis('off')
    else:
        pcm = ax[i].contourf(imgs[i, ...], cmap='RdBu_r', vmin=vmin, vmax=vmax, levels=50)
        ax[i].set_ylim(38, 0)
        ax[i].set_xlim(0, 10)
        ax[i].axis('off')

# %%
imgs = data_post_exp4[...,1,:,:].sum(axis=(-1,-2))
imgs = imgs - imgs.min()
vmin = imgs.min()
vmax = imgs.max()
fig, ax = plt.subplots(1, 6)
for i in range(6):
    if i == 5:
        plt.colorbar(pcm, ax=ax[i], aspect=30)
        ax[i].axis('off')
    else:
        pcm = ax[i].contourf(imgs[i, ...], cmap='Reds', vmin=vmin, vmax=vmax, levels=50)
        ax[i].set_ylim(38, 0)
        ax[i].set_xlim(0, 10)
        ax[i].axis('off')


# %%
