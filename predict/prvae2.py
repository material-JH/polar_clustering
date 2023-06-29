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
data_post_expm = np.load('output/set1_Ru_m002.npy')
data_post_exp = (data_post_exp - np.mean(data_post_exp)) / np.std(data_post_exp)
data_post_expm = (data_post_expm - np.mean(data_post_expm)) / np.std(data_post_expm)

#%%
data_post_exp = np.concatenate((data_post_exp, np.load('output/set1_Ru_m002.npy')), axis=0)
data_post_exp4 = np.load('output/set4_SRO_002.npy')
data_post_exp4 = np.concatenate((data_post_exp4, np.load('output/set4_SRO_m002.npy')), axis=0)

#%%

plt.imshow(data_post_expm[2].sum(axis=(-1, -2)))

#%%
mean = np.mean(data_post_exp)
std = np.std(data_post_exp)
data_post_exp = (data_post_exp - mean) / std

data_stack = data_post_exp.reshape(-1, 2, data_post_exp.shape[-2], data_post_exp.shape[-1])

mean4 = np.mean(data_post_exp4)
std4 = np.std(data_post_exp4)
data_post_exp4 = (data_post_exp4 - mean4) / std4

data_stack4 = data_post_exp4.reshape(-1, 2, data_post_exp4.shape[-2], data_post_exp4.shape[-1])
#%%

polarization_keys_exp = [torch.inf for _ in range(len(data_stack))]
polarization_keys_exp4 = [torch.inf for _ in range(len(data_stack4))]

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_sepm = np.load('output/disk_m002_dft.npz')
simulations_tot = np.stack(simulations_sep.values(), axis=0)
simulations_tot = np.stack((simulations_tot, np.stack(simulations_sepm.values(), axis=0)), axis=-3)
#%%
polarization_keys_sim = [float(k.split('_')[2]) for k in simulations_sep.keys()]
polarization_keys_sim = np.array(polarization_keys_sim)
p_mean = np.mean(polarization_keys_sim)
p_std = np.std(polarization_keys_sim)
polarization_keys_sim = (polarization_keys_sim - p_mean) / p_std
#%%
mean_sim = np.mean(simulations_tot)
std_sim = np.std(simulations_tot)
simulations_tot = (simulations_tot - mean_sim) / std_sim
#%%
input_dim = simulations_tot.shape[-2:]
#%%
filename = 'weights/pvae2_1.0_norm'
filename = 'weights/pvae2_1.0'
import gc
gc.collect()
rvae = prVAE2(256, input_dim, latent_dim=10,
                        numlayers_encoder=3, numhidden_encoder=256,
                        numlayers_decoder=3, numhidden_decoder=256,
                        p_weight=1e+2,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%
simulations_tot = fn_on_resized(simulations_tot, normalize_Data)
data_stack = fn_on_resized(data_stack, normalize_Data)
data_stack4 = fn_on_resized(data_stack4, normalize_Data)

# %%

sim_p = rvae.compute_p(torch.tensor(simulations_tot).to(rvae.device))

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
lim = 2
ax.hist2d(sim_p, polarization_keys_sim, bins=100)
ax.set_xlabel('Predicted polarization')
ax.set_ylabel('True polarization')
ax.set_title('Simulations')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
# %%
exp_p = rvae.compute_p(torch.tensor(data_stack4).to(rvae.device))

fig, ax = plt.subplots(1, 5, figsize=(5, 5))
lim = 1.5
for n, img in enumerate(exp_p.reshape(5, 38, 10)):
    pcm = ax[n].imshow(img, cmap='RdBu', interpolation='bessel', vmin=-lim, vmax=lim)
    # if n == 4:
    #     plt.colorbar(pcm, ax=ax[n])
    ax[n].axis('off')
    ax[n].set_title(f'{n - 2}V')
# %%

z_sim = rvae.encode(torch.tensor(simulations_tot[:,0]).to(rvae.device))[0]
# %%
