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
filename = 'weights/rvae_002_norm_47.tar'
rvae = regrVAE(input_dim, latent_dim=10,translation=True,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
#%%

if os.path.exists(filename):
    rvae.load_weights(filename)
    print('loaded weights')
#%%
rvae.save_model(filename.replace('weights/', 'models/'))
#%%
rvae = aoi.load_model(filename.replace('weights/', 'models/'))
#%%
ind_test = np.random.choice(range(len(imstack_train)), len(imstack_train) // 5, replace=False)
ind_train = np.setdiff1d(range(len(imstack_train)), ind_test)

#%%
rvae.fit(
    X_train= imstack_train[ind_train],
    X_test = imstack_train[ind_test],
    training_cycles=50,
    batch_size=2 ** 8,
    filename=filename)

#%%
rvae.fit(
    imstack_train, 
    training_cycles=10,
    batch_size=2 ** 8)

#%%
rvae.manifold2d(cmap='viridis', figsize=(10, 10), d=6)
#%%
encoded_mean, _ = rvae.encode(data_stack)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

encoded_mean2, _ = rvae.encode(data_stack2)
z21, z22, z23 = encoded_mean2[:,0], encoded_mean2[:, 1:3], encoded_mean2[:, 3:]
#%%
sim_mean, sim_sd  = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]
#%%
fig, ax = plt.subplots(1, 5, figsize=(15, 10))
for i, img in enumerate(encoded_mean[:1900,10].reshape(5, 38, 10)):
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
#%%

from cuml import TSNE

# tsne = UMAP(n_components=2, n_neighbors=15, min_dist=0.01, metric='euclidean', verbose=True, transform_seed=42)
tsne = TSNE(n_components=2, perplexity=30, verbose=True)
tsne.fit_transform(np.concatenate((z13, z33), axis=0))
#%%
from matplotlib.colors import LogNorm
plt.hist2d(tsne.embedding_[len(z13):,0], tsne.embedding_[len(z13):,1], bins=100, cmap='Blues', alpha=.5, norm=LogNorm())
# plt.hist2d(tsne.embedding_[:len(z13),0], tsne.embedding_[:len(z13),1], bins=100, cmap='Reds', alpha=.7, norm=LogNorm())

plt.tick_params(axis='both', direction='in')

#%%
np.savetxt('output/umap_002_sim.txt', tsne.embedding_[len(z13):])
np.savetxt('output/umap_002_exp.txt', tsne.embedding_[:len(z13)])
#%%
plt.scatter(tsne.embedding_[len(z13):,0], tsne.embedding_[len(z13):,1], alpha=.1, color='blue', label='sim')
plt.scatter(tsne.embedding_[:len(z13),0], tsne.embedding_[:len(z13),1], alpha=.1, color='red', label='exp')
#%%
#%%

plt.scatter(z13[:,0], z13[:,1], alpha=.1, color='blue', label='exp')
plt.scatter(z33[:,0], z33[:,1], alpha=.1, color='red', label='sim')
#%%
plt.hist2d(z11[:1900], z13[:1900,0], bins=100, cmap='Reds', alpha=.5, norm=LogNorm())
plt.hist2d(z31, z33[:,0], bins=100, cmap='Blues', alpha=.7, norm=LogNorm())
#%%

# norm_p = polarization_keys - polarization_keys.min()
# norm_p = norm_p / norm_p.max()
plt.scatter(z13[:,0], z13[:,1], alpha=.1, color='red', label='exp')
# plt.scatter(z23[:,0], z23[:,1], alpha=.1, color='yellow', label='exp2')
plt.scatter(z33[:,0], z33[:,1], alpha=.1, color='blue', label='sim')
plt.colorbar()
plt.legend()
plt.xlim(-2, 2)
plt.rcParams.update({'font.size': 20})
#%%

plt.imshow(alpha[:1900,0].reshape(5, 38, 10)[2,:,:], cmap='gray')
plt.colorbar()
# %%

mid = np.zeros((10, 38, 10))
mid[:, 10:25, :] = 1
mid = mid.flatten()
#%%
xyz = reduce((lambda x, y: x * y), data_post_exp.shape[:3])
# xyz //= 2
tot = 0
for n in range(xyz):
    if mid[n] == 0:
        continue
    distances = np.linalg.norm(z33 - z13[n], axis=1)
    # distances += np.abs(z31 - z11[n])
    # distances += np.linalg.norm(z33 - z13[n + xyz], axis=1)
    # distances += np.abs(z31 - z11[n + xyz])
    if np.min(distances) > 0.2:
        continue
    tot += 1
    if tot > 10:
        break
    nearest_neighbor_index = np.argmin(distances)
    fig, ax = plt.subplots(2, 2, figsize=(5,5))
    # fig.suptitle(f'{n} {nearest_neighbor_index}')
    fig.suptitle('original vs decoded')
    ax[0,1].imshow(rvae.decode(np.array([*z13[n], *alpha[n]]))[0])
    ax[0,1].axis('off')
    ax[1,1].imshow(rvae.decode(np.array([*z33[nearest_neighbor_index], *alpha_sim[nearest_neighbor_index]]))[0])
    ax[1,1].axis('off')
    ax[0,0].imshow(data_stack[n])
    ax[0,0].axis('off')
    ax[1,0].imshow(simulations_tot[nearest_neighbor_index])
    ax[1,0].axis('off')
    plt.show()
# %%
xyz = reduce((lambda x, y: x * y), data_post_exp.shape[:3])

is_closed = []
for n in range(xyz):
    distances = np.linalg.norm(z33 - z13[n], axis=1)
    # distances = np.linalg.norm(z33 - z13[n], axis=1) + abs(z31 - z11[n]) / 10
    if np.min(distances) > .1:
        is_closed.append(0)
    else:
        is_closed.append(1)

is_closed = np.array(is_closed)

fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(8, 4))
for n, img in enumerate(is_closed.reshape(data_post_exp.shape[:3])):
    im = axs[n].imshow(img)
    axs[n].axis('off')
plt.show()
#%%

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 6))
for n, img in enumerate(z13[:1900].reshape((*data_post_exp[:5].shape[:3], -1))):
    img = img - img.min()
    img = img / img.max()
    img = 1 - img
    img = img[:,:,1:4]
    # img = np.stack([img[:,:,0], np.ones(img.shape[:2]), img[:,:,1]])
    img = np.concatenate([img, np.ones((*img.shape[:2], 1))], axis=-1)
    # img = np.moveaxis(img, 0, -1)
    axs[n].imshow(img)
    # im1 = axs[n].imshow(img[:,:,0], cmap='Reds', origin='lower', alpha=0.5, label='Channel 1',vmin=-1, vmax=1)
    # im2 = axs[n].imshow(img[:,:,1], cmap='Blues', origin='lower', alpha=0.5, label='Channel 2',vmin=-1, vmax=1)
    # im3 = axs[n].imshow(im[1], cmap='Greens', origin='lower', alpha=0.3, label='Channel 2',vmin=-1, vmax=4)
    axs[n].axis('off')
plt.show()

# %%
tot = np.concatenate((imstack_train, simulations_tot), axis=0)
means = []
stds = []
for i in range(tot.shape[0]):
    means.append(tot[i].mean())
    stds.append(tot[i].std())
fig, ax = plt.subplots()
ax.errorbar(range(tot.shape[0]), means, yerr=stds, fmt='o', capsize=5)

# %%
polarization_keys = [float(k.split('_')[2]) for k in simulations_sep.keys()]

xyz = reduce((lambda x, y: x * y), data_post_exp[:5].shape[:3])
new_dat = np.zeros(xyz)

min_indexes = []

for n in range(xyz):
    distances = np.linalg.norm(z33[:len(polarization_keys)] - z13[n], axis=1)
    # distances += abs(z31[:len(polarization_keys)] - z11[n])
    distances += np.linalg.norm(z33[len(polarization_keys):] - z13[n + xyz], axis=1)
    # distances += abs(z31[len(polarization_keys):] - z11[n + xyz])

    nearest_neighbor_index = np.argmin(distances)

    min_indexes.append(nearest_neighbor_index)
    new_dat[n] = polarization_keys[nearest_neighbor_index]
#%%
amap = alpha[:1900,3].reshape(5, 38, 10)
# amap = [int(x == 1 or y == 1) for x, y in zip(alpha[:1900,3], alpha[:1900,4])]
amap = np.array(amap).reshape(5, 38, 10)
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 10))
for n, img in enumerate(new_dat.reshape(data_post_exp[:5].shape[:3])):
    # img[np.where(amap[n] < 0.5)] = 0
    im = axs[n].imshow(img, cmap='bwr', aspect='auto', interpolation='bicubic')
    # axs[n].imshow(amap[n], cmap='gray', aspect='auto', interpolation='bicubic', alpha=(1 - amap[n])/ 2)
    axs[n].axis('off')
    # if n == 9:
    #     fig.colorbar(im, ax=axs[n], orientation='vertical', fraction=.2)

# cbar = fig.colorbar(im)
plt.show()

#%%

def macroscopic_average(potential, period_points):
    """Getting the macroscopic average of potential
    Args:
        potential : array containig the electrostaticpotential/charge density
        periodicity : real number; the period over which to average
        resolution : the grid resolution in the direction of averaging
    Returns:
        macro_average : array with the macroscopically averaged values"""


    macro_average = np.zeros(shape=(len(potential)))
    # Period points must be even
    if period_points % 2 != 0:
        period_points = period_points + 1

    length = len(potential)
    for i in range(length):
        start = i - int(period_points / 2)
        end = i + int(period_points / 2)
        if start < 0:
            start = start + length
            macro_average[i] = macro_average[i] + sum(potential[0:end]) + sum(potential[start:length])
            macro_average[i] = macro_average[i] / period_points
        elif end >= length:
            end = end - length
            macro_average[i] = macro_average[i] + sum(potential[start:length]) + sum(potential[0:end])
            macro_average[i] = macro_average[i] / period_points
        else:
            macro_average[i] = macro_average[i] + sum(potential[start:end]) / period_points

    return macro_average

from scipy.interpolate import make_interp_spline
x = np.linspace(0, 37, 38)
xnew = np.linspace(0, 37, 100)
line = new_dat.reshape(data_post_exp[:5].shape[:3])
line = np.mean(line, axis=2)

fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 10))
for n, l in enumerate(line):
    l = macroscopic_average(l, 5)
    spline = make_interp_spline(x, l)
    ynew = spline(xnew)
    axs[n].plot(xnew, ynew)
    axs[n].set_ylim(-.015, .015)
    axs[n].set_xlim(0, 37)
    axs[n].plot([0, 37], [0, 0], 'k--')

#%%
from scipy.interpolate import interp1d
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(12, 5))
outputs = []
for n, img in enumerate(new_dat.reshape(data_post_exp.shape[:3])):
    line = axs[n].fill_betweenx(np.mean(img, axis=1), range(img.shape[0]), where=np.mean(img, axis=1)<0, color='green', alpha=0.3, interpolate=True)
    f = interp1d(np.linspace(0, img.shape[0] - 1, img.shape[0]), np.mean(img, axis=1), kind='cubic')
    outputs.append(f(np.linspace(0, img.shape[0] - 1, img.shape[0] * 10)))
    # outputs.append(np.mean(img, axis=1))
    axs[n].set_yticks([])
    axs[n].set_ylim(img.shape[0], 0)
    # if n == 9:
    #     fig.colorbar(im, ax=axs[n], orientation='vertical', fraction=.2)
plt.show()

np.savetxt(X=np.transpose(outputs), fname='outputs.csv', delimiter='\t')


#%%
n = 6
plt.fill_between(range(img.shape[0]), outputs[n], where=outputs[n]<0, color='grey', alpha=0.3, interpolate=True)
plt.fill_between(range(img.shape[0]), outputs[n], where=outputs[n]>0, color='red', alpha=0.3, interpolate=True)
# %%

def plot_vertical(data, figsize=(8, 12)):
    shape = data.shape
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize)
    for i in range(shape[0]):
        for j in range(shape[1]):
            axs[i, j].imshow(data[i, j])
            axs[i, j].axis('off')
    plt.show()

plot_vertical(data_post_exp[2][::4,::2], figsize=(5, 10))
# %%
