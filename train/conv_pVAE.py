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
data_post_exp = np.concatenate((data_post_exp, 
                                fn_on_resized(np.load('output/set1_Ru_m002.npy'), imutils.rotate, 180)), axis=0)
data_post_exp4 = np.load('output/set4_SRO_002.npy')
data_post_exp4 = np.concatenate((data_post_exp4, 
                                fn_on_resized(np.load('output/set4_SRO_m002.npy'), imutils.rotate, 180)), axis=0)

#%%

plt.imshow(data_post_exp4[2].sum(axis=(-1, -2)))

#%%
data_stack = data_post_exp.reshape(-1, 2, data_post_exp.shape[-2], data_post_exp.shape[-1])
data_stack4 = data_post_exp4.reshape(-1, 2, data_post_exp4.shape[-2], data_post_exp4.shape[-1])
#%%

polarization_keys_exp = [torch.inf for _ in range(len(data_stack))]
polarization_keys_exp4 = [torch.inf for _ in range(len(data_stack4))]

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_sepm = np.load('output/disk_m002_dft.npz')
simulations_tot = np.stack(simulations_sep.values(), axis=0)
simulations_tot = np.stack((simulations_tot, fn_on_resized(np.stack(simulations_sepm.values(), axis=0), imutils.rotate, 180)), axis=-3)
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
filename = 'weights/conv_pvae2_1.0_norm'
import gc
gc.collect()
rvae = pVAE2(input_dim, latent_dim=10, h_dim=128,
                        p_weight=5e+2,
                        filename=filename)

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
z, _ = rvae.encode(data_stack4[:,1])
decoded = rvae.decode(z)
#%%
fig, ax = plt.subplots(1, 2)
ind = 100
ax[0].imshow(data_stack4[ind,1])
ax[1].imshow(decoded[ind])
# %%
rvae.compute_p(imstack_train[ind_test])
# %%
self = rvae
x2 = data_stack4
with torch.no_grad():
    z2d_mean = torch.zeros((x2.shape[0], (self.z_dim) * 2)).to(self.device)
    for i in range(2):
        x = x2[:,i,:,:]
        z_mean, z_logsd = self.encode(x)
        z2d_mean[:, i * self.z_dim:(i + 1) * self.z_dim] = torch.tensor(z_mean)
    p = self.fcn_net(z2d_mean)

# %%
plt.hist2d(p.cpu().numpy(), polarization_keys_sim, bins=100)
# %%

decoded = rvae.decode(z_mean)
# %%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(x[400])
ax[1].imshow(imutils.rotate(x[400], 180))
# %%
