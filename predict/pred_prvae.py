#%%
import os
os.chdir('../')

import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from model.model import *
from lib.plot import *
#%%
data_post_exp = np.load('output/set1_Ru_002.npy')

# data_post_exp2 = np.load('output/set2_Ru_002.npy')
# data_post_exp3 = np.load('output/set3_SRO_002.npy')
data_post_exp4 = np.load('output/set4_SRO_002.npy')

#%%

mean = np.mean(data_post_exp)
std = np.std(data_post_exp)
data_post_exp = (data_post_exp - mean) / std

data_stack = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])

mean4 = np.mean(data_post_exp4)
std4 = np.std(data_post_exp4)
data_post_exp4 = (data_post_exp4 - mean4) / std4

data_stack4 = data_post_exp4.reshape(-1, data_post_exp4.shape[-2], data_post_exp4.shape[-1])

#%%

data_stack = fn_on_resized(data_stack, imutils.resize, 64, 64)
data_stack4 = fn_on_resized(data_stack4, imutils.resize, 64, 64)
#%%

data_stack = data_stack[...,None]
data_stack4 = data_stack4[...,None]

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_sepm = np.load('output/disk_m002_dft.npz')
#%%
simulations_tot = np.stack(simulations_sep.values(), axis=0)
simulations_tot = fn_on_resized(simulations_tot, imutils.resize, 64, 64)
simulations_tot = simulations_tot[...,None]
# simulations_tot = np.stack((simulations_tot, np.stack(simulations_sepm.values(), axis=0)), axis=-3)
#%%
polarization_keys_sim = [float(k.split('_')[2]) for k in simulations_sep.keys()]
polarization_keys_sim = np.array(polarization_keys_sim)
p_mean = np.mean(polarization_keys_sim)
p_std = np.std(polarization_keys_sim)
polarization_keys_sim = (polarization_keys_sim - p_mean) / p_std
#%%
mean3 = np.mean(simulations_tot)
std3 = np.std(simulations_tot)
simulations_tot = (simulations_tot - mean3) / std3
#%%
# Intitialize rVAE model

input_dim = data_stack4.shape[1:]
#%%
filename = 'weights/conv_prvae_002_norm_10'
import gc
gc.collect()
pconv_vae = prVAE(input_dim, latent_dim=10,
                        p_weight=1e+3,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)

enet = EncoderNet(1, 128, latent_dim=13,  kernel_size=5, stride=1, padding=2)
# dnet = DecoderNet(10, 128)
# pconv_vae.set_model(enet, dnet)
pconv_vae.set_encoder(enet)
#%%
if os.path.exists(f'{filename}.tar'):
    pconv_vae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%
gc.collect()


def pred_p(self, x: torch.Tensor | np.ndarray):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if len(x.shape) == 3:
        x = x.unsqueeze(-1)
    x = x.to(self.device)
    with torch.no_grad():
        z_mean, z_logsd = self.encode(x)
        p = self.fcn_net(torch.tensor(z_mean).to(self.device))
    return p


exp_p = pred_p(pconv_vae, data_stack4)
#%%
with torch.no_grad():
    selected_inds = np.random.choice(range(len(simulations_tot)), 4, replace=False)
    selected_img = simulations_tot[selected_inds]
    print(selected_inds)
    z_mean, z_logsd = pconv_vae.encode(torch.tensor(selected_img).float().to(pconv_vae.device))
    decoded = pconv_vae.decode(z_mean[:,3:])
    vmin = np.min(selected_img)
    vmax = np.max(selected_img) * 0.8
    fig, ax = plt.subplots(2, 2, figsize=(4,4))
    for i in range(2):
        for j in range(2):
            ax[i,j].imshow(selected_img[i*2+j].reshape(64,64), vmin=vmin, vmax=vmax)
            ax[i,j].axis('off')
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(2, 2, figsize=(4,4))
    for i in range(2):
        for j in range(2):
            ax[i,j].imshow(decoded[i*2+j].reshape(64,64), vmin=vmin, vmax=vmax)
            ax[i,j].axis('off')
    plt.tight_layout()
    plt.show()
    print(pred_p(pconv_vae, selected_img) * p_std + p_mean)
#%%
# exp_p = rvae.compute_p(torch.tensor(data_stack4).to(rvae.device))
exp_p = exp_p * p_std + p_mean
exp_p = exp_p.cpu().detach().numpy()
exp_p = exp_p.reshape(data_post_exp.shape[:-2])
#%%
fig, ax = plt.subplots(1, 5, figsize=(5,5))
for i in range(5):
    ax[i].imshow(exp_p[i], cmap='RdBu', vmin=-.02, vmax=.02)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

# %%
with torch.no_grad():
    z_mean, z_logsd = pconv_vae.encode(torch.tensor(data_stack4).float().to(pconv_vae.device))

#%%
fig, ax = plt.subplots(1, pconv_vae.z_dim - 3, figsize=(5,5))
for i in range(pconv_vae.z_dim - 3):
    ax[i].imshow(z_mean[:,i].reshape(data_post_exp.shape[:-2])[2], cmap='RdBu')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
# %%
