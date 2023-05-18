#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi

#%%
data_post_exp = np.load('output/set1_SRO_002.npy')
data_post_exp2 = np.load('output/set4_Ru_002.npy')
data_post_exp = np.concatenate((data_post_exp, data_post_exp2), axis=0)
#%%
data_post_exp = fn_on_resized(data_post_exp, normalize_Data)
# %%
imstack_train = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])
#%%
import cv2
simulations_sep = np.load('output/disk_002_dft.npz')
simulations = {}
for k, v in simulations_sep.items():
    simulations[k] = v

for k, v in simulations.items():
    simulations[k] = normalize_Data(simulations[k])
simulations_tot = np.stack(list(simulations.values()), axis=0)
#%%
polarization_keys = [float(k.split('_')[3]) for k in simulations_sep.keys()]
polarization_keys = np.array(polarization_keys)[None,:]
print(polarization_keys.shape)
#%%
# Intitialize rVAE model
input_dim = imstack_train.shape[1:]
rvae = aoi.models.rVAE(input_dim, latent_dim=2,nb_classes=1,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,)

# if os.path.exists('output/rvae_002_norm.tar'):
#     rvae.load_weights('output/rvae_002_norm.tar')
#     print('loaded weights')
rvae.fit(X_train=simulations_tot,
    y_train=polarization_keys,
    training_cycles=10,
    batch_size=2 ** 8)

# %%
