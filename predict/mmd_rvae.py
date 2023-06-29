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

plt.imshow(data_post_exp4[2].sum(axis=(-1, -2)))

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
mean_sim = np.mean(simulations_tot)
std_sim = np.std(simulations_tot)
simulations_tot = (simulations_tot - mean_sim) / std_sim
#%%
# Intitialize rVAE model

imstack_train = np.concatenate((data_stack, simulations_tot), axis=0)
imstack_train = np.concatenate((imstack_train, data_stack4), axis=0)
imstack_train = fn_on_resized(imstack_train, normalize_Data)
# imstack_train = np.concatenate((imstack_train, simulations_tot), axis=0)
imstack_polar = np.concatenate((polarization_keys_exp, polarization_keys_sim), axis=0)
imstack_polar = np.concatenate((imstack_polar, polarization_keys_exp4), axis=0)
# imstack_polar = np.concatenate((imstack_polar, polarization_keys_sim), axis=0)
input_dim = imstack_train.shape[-2:]
#%%
filename = 'weights/mmdrvae_1.1_norm'
import gc
gc.collect()
rvae = MMDrVAE(input_dim, latent_dim=10,
                        numlayers_encoder=3, numhidden_encoder=256,
                        numlayers_decoder=3, numhidden_decoder=256,
                        p_weight=1e+2,
                        filename=filename)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%
sim_p = rvae.compute_p(simulations_tot)
# %%
plt.hist2d(sim_p, polarization_keys_sim, bins=50)
# %%
exp_p = rvae.compute_p(fn_on_resized(data_stack4, normalize_Data))

# %%
imgs = exp_p.reshape(data_post_exp.shape[:-2])
vmin, vmax = -1, 1
fig, ax = plt.subplots(1, 5, figsize=(10, 5))
for i in range(5):
    ax[i].imshow(imgs[i], vmin=vmin, vmax=vmax, cmap='RdBu_r')

# %%
with open(filename + '_plosses.txt', 'r') as f:
    plosses = np.array([float(line) for line in f.readlines()])
    plt.plot(plosses)
# %%
