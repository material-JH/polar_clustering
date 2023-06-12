#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from model.model import *
#%%
data_post_exp = np.load('output/set1_Ru_002.npy')
data_post_exp = np.concatenate((data_post_exp, np.load('output/set1_Ru_m002.npy')), axis=0)
data_post_exp2 = np.load('output/set4_SRO_002.npy')
data_post_exp2 = np.concatenate((data_post_exp2, np.load('output/set4_SRO_m002.npy')), axis=0)

#%%

mean = np.mean(data_post_exp)
std = np.std(data_post_exp)
data_post_exp = (data_post_exp - mean) / std
mean2 = np.mean(data_post_exp2)
std2 = np.std(data_post_exp2)
data_post_exp2 = (data_post_exp2 - mean2) / std2

data_stack = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])
data_stack2 = data_post_exp2.reshape(-1, data_post_exp2.shape[-2], data_post_exp2.shape[-1])

#%%
simulations_sep = np.load('output/disk_002_dft.npz')
simulations_sepm = np.load('output/disk_m002_dft.npz')
simulations_tot = np.stack(simulations_sep.values(), axis=0)
simulations_tot = np.concatenate((simulations_tot, np.stack(simulations_sepm.values(), axis=0)), axis=0)
#%%

mean3 = np.mean(simulations_tot)
std3 = np.std(simulations_tot)
simulations_tot = (simulations_tot - mean3) / std3
#%%
# Intitialize rVAE model

imstack_train = np.concatenate((data_stack, simulations_tot), axis=0)

input_dim = imstack_train.shape[1:]

#%%
rvae = regrVAE(input_dim, latent_dim=4,
                        numlayers_encoder=2, numhidden_encoder=256,
                        numlayers_decoder=2, numhidden_decoder=256,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=1, div_weight=1, cont_weight=1,)
#%%


if os.path.exists('weights/regrvae_002_norm_47_256.tar'):
    rvae.load_weights('weights/regrvae_002_norm_47_256.tar')
    print('loaded weights')
# %%
encoded_mean, _ = rvae.encode(data_stack)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

encoded_mean2, _ = rvae.encode(data_stack2)
z21, z22, z23 = encoded_mean2[:,0], encoded_mean2[:, 1:3], encoded_mean2[:, 3:]
#%%
sim_mean, sim_sd = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]
#%%
fig, ax = plt.subplots(1, 5, figsize=(15, 10))
for i, img in enumerate(z13[:1900,0].reshape(5, 38, 10)):
    ax[i].imshow(img, cmap='RdBu')
    ax[i].axis('off')
#%%

output = {}

for n, (k, v, v2) in enumerate(zip(simulations_sep.keys(), sim_mean[:len(simulations_sep)], sim_mean[len(simulations_sep):])):
    # if n in min_indexes:
    output[k] = np.stack((v[[0, *range(3, len(sim_mean[0]))]], v2[[0, *range(3, len(sim_mean[0]))]]), axis=0)

print(len(output))
np.savez('output/z3', **output)

#%%

data_post_p = np.load('output/set1_Ru_002.npy')
data_post_m = np.load('output/set1_Ru_m002.npy')

data_post_p = fn_on_resized(data_post_p, normalize_Data)
data_post_m = fn_on_resized(data_post_m, normalize_Data)

stack_p = data_post_p.reshape(-1, data_post_p.shape[-2], data_post_p.shape[-1])
stack_m = data_post_m.reshape(-1, data_post_m.shape[-2], data_post_m.shape[-1])

output = []

for v, v2 in zip(stack_p, stack_m):
    output.append(np.stack((rvae.encode(normalize_Data(v))[0][0][[0, *range(3, len(sim_mean[0]))]], rvae.encode(normalize_Data(v2))[0][0][[0, *range(3, len(sim_mean[0]))]]), axis=0))

np.save('output/z1', output)
#%%

data_post_p = np.load('output/set4_SRO_002.npy')
data_post_m = np.load('output/set4_SRO_m002.npy')

data_post_p = fn_on_resized(data_post_p, normalize_Data)
data_post_m = fn_on_resized(data_post_m, normalize_Data)

stack_p = data_post_p.reshape(-1, data_post_p.shape[-2], data_post_p.shape[-1])
stack_m = data_post_m.reshape(-1, data_post_m.shape[-2], data_post_m.shape[-1])

output = []

for v, v2 in zip(stack_p, stack_m):
    output.append(np.stack((rvae.encode(normalize_Data(v))[0][0][[0, *range(3, len(sim_mean[0]))]], rvae.encode(normalize_Data(v2))[0][0][[0, *range(3, len(sim_mean[0]))]]), axis=0))

np.save('output/z2', output)

#%%
arr = np.array(list(output.values()))
print(arr.shape)