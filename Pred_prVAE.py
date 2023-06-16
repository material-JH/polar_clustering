#%%
import sys
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
polarization_keys_exp = [torch.inf for _ in range(len(data_stack))]
polarization_keys_exp4 = [torch.inf for _ in range(len(data_stack4))]

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_sepm = np.load('output/disk_m002_dft.npz')
simulations_tot = np.stack(simulations_sep.values(), axis=0)
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

imstack_train = np.concatenate((data_stack, simulations_tot), axis=0)
imstack_train = np.concatenate((imstack_train, data_stack4), axis=0)
# imstack_train = np.concatenate((imstack_train, simulations_tot), axis=0)
imstack_polar = np.concatenate((polarization_keys_exp, polarization_keys_sim), axis=0)
imstack_polar = np.concatenate((imstack_polar, polarization_keys_exp4), axis=0)
# imstack_polar = np.concatenate((imstack_polar, polarization_keys_sim), axis=0)
input_dim = imstack_train.shape[-2:]
# %%
filename = 'weights/prvae_002_norm_10_3_pnu'
import gc
gc.collect()
rvae = prVAE(input_dim, latent_dim=10,
                        numlayers_encoder=2, numhidden_encoder=256,
                        numlayers_decoder=2, numhidden_decoder=256,
                        p_weight=1e+3,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
# %%

pred_p = rvae.compute_p(torch.tensor(simulations_tot).to(rvae.device))
pred_p = pred_p * p_std + p_mean

# %%
plt.hist2d(polarization_keys_sim * p_std + p_mean, pred_p.cpu().detach().numpy(), bins=50)
# %%
exp_p = rvae.compute_p(torch.tensor(data_stack).to(rvae.device))
# exp_p = rvae.compute_p(torch.tensor(data_stack4).to(rvae.device))
fig, ax = plt.subplots(1, 5, figsize=(5,5))
exp_p = exp_p * p_std + p_mean
exp_p = exp_p.cpu().detach().numpy()
exp_p = exp_p.reshape(data_post_exp.shape[:-2])
for i in range(5):
    ax[i].imshow(exp_p[i], cmap='RdBu')

# %%
