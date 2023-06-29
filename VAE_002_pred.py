#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from model.model import *
from lib.plot import *
#%%
data_post_exp = np.load('output/set1_Ru_002.npy')
data_post_exp = np.concatenate((data_post_exp, np.load('output/set1_Ru_m002.npy')), axis=0)
#%%
data_post_exp2 = np.load('output/set4_SRO_m002.npy')
# data_post_exp2 = np.vstack((data_post_exp2, np.load('output/set4_SRO_m002.npy')))
#%%
arr = data_post_exp2.sum(axis=(-1, -2))
arr = arr - np.min(arr)
arr = arr / np.max(arr)
plt.imshow(arr[2])

#%%

fig, ax = plt.subplots(1,4, figsize=(6, 5))
# vmax = np.max(output)
# vmin = np.min(output)
vmax = 1
vmin = 0
# output[lbl==0] = 0
ylim = np.array([34, 1])
# ylim = [34, 2]
for n, img in enumerate(arr[::-1]):
    # zero_img = np.zeros_like(img)
    # zero_img[np.logical_and(-0.01 < img,  img < 0.01)] = 1
    # print(img.min(), img.max())
    if n % 2 == 1:
        continue
    n = n // 2

    ax[n].contourf(img, alpha=0.6, cmap='Reds', vmin=vmin, vmax=vmax, levels=50)
    ax[n].set_ylim(*ylim)
    if n == 0:
        ylim -= 1
        ax[n].set_ylim(*ylim)
        ylim += 1
        # ax[n].set_ylim(-2, 38 - 2)
    # if n == 4:
    #     plt.colorbar(pcm, ax=ax[n], ticks=[vmin, 0, vmax])
    # ax[n].imshow(zero_img, cmap='gray_r')

    ax[n].set_xticks([])
    ax[n].set_yticks([])
    x = range(38)
    if n == 0:
        x = range(2, 40)
        c = 'b'
    elif n == 1:
        c = 'gray'
    elif n == 2:
        c = 'r'
    ax[-1].plot(np.mean(img, axis=1), x, c=c, alpha=.5)

ax[-1].set_ylim(*ylim)
# ax[-1].set_ylim(38, 0)
ax[-1].set_xticks([])
ax[-1].set_yticks([])
# ax[-1].axis('off')
# fig.colorbar(pcm, ax=ax[-1])
plt.show()

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
filename = 'weights_pnu/rvae_002_norm_47.tar'
rvae = regrVAE(input_dim, latent_dim=10,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
#%%

if os.path.exists(filename + '.tar'):
    rvae.load_weights(filename + '.tar')
    print('loaded weights')
#%%

# %%
encoded_mean, _ = rvae.encode(data_stack)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

encoded_mean2, _ = rvae.encode(data_stack2)
z21, z22, z23 = encoded_mean2[:,0], encoded_mean2[:, 1:3], encoded_mean2[:, 3:]
#%%
sim_mean, sim_sd = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]
#%%
for j in range(10):
    fig, ax = plt.subplots(1, 5, figsize=(15, 10))
    vmin = np.min(z13[1900:,j])
    vmax = np.max(z13[1900:,j])
    for i, img in enumerate(z23[1900:,j].reshape(5, 38, 10)):
        ax[4 - i].imshow(img, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax[4 - i].axis('off')
    plt.show()

#%%

fig, ax = plt.subplots(1, 5, figsize=(15, 10))
for i, img in enumerate(data_post_exp[:5].sum(axis=(-1, -2)).reshape(5, 38, 10)):
    ax[4 - i].imshow(img, cmap='RdBu')
    ax[4 - i].axis('off')

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
    output.append(np.stack((rvae.encode(normalize_Data(v))[0][0][[0, *range(3, len(encoded_mean[0]))]], rvae.encode(normalize_Data(v2))[0][0][[0, *range(3, len(encoded_mean[0]))]]), axis=0))

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
    output.append(np.stack((rvae.encode(normalize_Data(v))[0][0][[0, *range(3, len(encoded_mean[0]))]], rvae.encode(normalize_Data(v2))[0][0][[0, *range(3, len(encoded_mean[0]))]]), axis=0))

np.save('output/z2', output)

#%%
arr = np.array(list(output.values()))
print(arr.shape)