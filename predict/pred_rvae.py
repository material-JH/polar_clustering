#%%
import imutils
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from lib.main import *
import atomai as aoi
from model.model_pnu import *
#%%
data_post_exp = np.load('../output/set1_Ru_002.npy')

# data_post_exp2 = np.load('output/set2_Ru_002.npy')
# data_post_exp3 = np.load('output/set3_SRO_002.npy')
data_post_exp4 = np.load('../output/set4_SRO_002.npy')

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
simulations_sep = np.load('../output/disk_002_dft.npz')
# simulations_sepm = np.load('output/disk_m002_dft.npz')
# simulations_tot = np.stack(simulations_sep.values(), axis=0)
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
imstack_train = normalize_Data(imstack_train)
tmp = np.zeros((len(imstack_train), 64, 64))
for i in range(len(imstack_train)):
    tmp[i] = imutils.resize(imstack_train[i], 64, 64)
#%%
imstack_train = tmp[...,None]

# imstack_train = np.concatenate((imstack_train, simulations_tot), axis=0)
imstack_polar = np.concatenate((polarization_keys_exp, polarization_keys_sim), axis=0)
imstack_polar = np.concatenate((imstack_polar, polarization_keys_exp4), axis=0)
# imstack_polar = np.concatenate((imstack_polar, polarization_keys_sim), axis=0)

input_dim = imstack_train.shape[1:]
#%%
filename = '../weights/conv_prvae_002_norm_10'
import gc
gc.collect()
pconv_vae = prVAE((64, 64, 1), latent_dim=10,
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
ind_test = np.random.choice(range(len(imstack_train)), len(imstack_train) // 5, replace=False)
ind_train = np.setdiff1d(range(len(imstack_train)), ind_test)

#%%
with torch.no_grad():
    tmp = np.zeros((len(simulations_tot), 64, 64))
    for i in range(len(simulations_tot)):
        tmp[i] = imutils.resize(simulations_tot[i], 64, 64)
    z_mean, z_logsd = pconv_vae.encode(torch.tensor(tmp).float().unsqueeze(1).to(pconv_vae.device))
    p = pconv_vae.fcn_net(torch.tensor(z_mean).to(pconv_vae.device))

# %%
plt.hist2d(polarization_keys_sim, p.cpu().detach().numpy(), bins=50)
# %%
with torch.no_grad():
    tmp = np.zeros((len(data_stack4), 64, 64))
    for i in range(len(data_stack4)):
        tmp[i] = imutils.resize(data_stack4[i], 64, 64)

    z_mean, z_logsd = pconv_vae.encode(torch.tensor(tmp).float().unsqueeze(1).to(pconv_vae.device))
    exp_p = pconv_vae.fcn_net(torch.tensor(z_mean).to(pconv_vae.device))


# exp_p = rvae.compute_p(torch.tensor(data_stack4).to(rvae.device))
fig, ax = plt.subplots(1, 5, figsize=(5,5))
exp_p = exp_p * p_std + p_mean
exp_p = exp_p.cpu().detach().numpy()
exp_p = exp_p.reshape(data_post_exp.shape[:-2])
for i in range(5):
    ax[i].imshow(exp_p[i], cmap='RdBu', vmin=-2e-2, vmax=2e-2)

# %%
fig, ax = plt.subplots(1, 5, figsize=(5,5))
imgs = z_mean[:,12].reshape(data_post_exp.shape[:-2])
for i in range(5):
    ax[i].imshow(imgs[i], cmap='RdBu')

# %%
