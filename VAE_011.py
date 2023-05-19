#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from lib.plot import *
#%%
data_post_exp = np.load('output/set1_Ru_011.npy')
data_post_exp = np.concatenate((data_post_exp, np.load('output/set1_Ru_m011.npy')), axis=0)
data_post_exp2 = np.load('output/set4_SRO_011.npy')
data_post_exp2 = np.concatenate((data_post_exp2, np.load('output/set4_SRO_m011.npy')), axis=0)
#%%
data_post_exp = fn_on_resized(data_post_exp, normalize_Data)
data_post_exp2 = fn_on_resized(data_post_exp2, normalize_Data)

data_stack = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])
data_stack2 = data_post_exp2.reshape(-1, data_post_exp2.shape[-2], data_post_exp2.shape[-1])

#%%
simulations_sep = np.load('output/disk_011_dft.npz')
simulations_sepm = np.load('output/disk_m011_dft.npz')
simulations_tot = np.stack(list(map(normalize_Data, simulations_sep.values())), axis=0)
simulations_tot = np.concatenate((simulations_tot, np.stack(list(map(normalize_Data, simulations_sepm.values())), axis=0)), axis=0)
#%%
imstack_train = np.concatenate((data_stack, simulations_tot), axis=0)
input_dim = imstack_train.shape[1:]
rvae = aoi.models.jrVAE(input_dim, latent_dim=47,
                        numlayers_encoder=2, numhidden_encoder=256,
                        numlayers_decoder=2, numhidden_decoder=256,
                        discrete_dim=[2] * 4)
#%%
rvae = aoi.load_model('models/rvae_011_norm_47_256.tar')
#%%
ind_train = np.random.choice(range(len(imstack_train)), len(imstack_train), replace=False)
ind_test = np.random.choice(ind_train, len(imstack_train) // 5, replace=False)
ind_train = np.setdiff1d(ind_train, ind_test)

#%%
rvae.fit(
    X_train= imstack_train[ind_train],
    X_test = imstack_train[ind_test],
    training_cycles=150,
    batch_size=2 ** 8,
    filename='weights/rvae_011_norm_47_256')
rvae.save_model('models/rvae_011_norm_47_256')
#%%
encoded_mean, _ , alpha = rvae.encode(data_stack)
z11, z12, z13 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

encoded_mean2, _ , alpha2 = rvae.encode(data_stack2)
z21, z22, z23 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]
#%%
sim_mean, sim_sd, alpha_sim  = rvae.encode(simulations_tot)
z31, z32, z33 = sim_mean[:,0], sim_mean[:, 1:3], sim_mean[:, 3:]
#%%

#%%

output = {}

for n, (k, v, v2) in enumerate(zip(simulations_sep.keys(), sim_mean[:len(simulations_sep)], sim_mean[len(simulations_sep):])):
    if n in min_indexes:
        output[k] = np.stack((v[[0, *range(3, len(sim_mean[0]))]], v2[[0, *range(3, len(sim_mean[0]))]]), axis=0)

print(len(output))
np.savez('output/z33', **output)

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

np.save('output/z13', output)

#%%
arr = np.array(list(output.values()))
print(arr.shape)
#%%
plt.scatter(z13[:,0], z13[:,1], alpha=.1, color='blue', label='exp')
plt.scatter(z33[:,0], z33[:,1], alpha=.1, color='red', label='sim')
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
db_cluster = DBSCAN(eps=0.12, min_samples=3).fit(z13)
print(db_cluster.labels_)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8, 4))
for n, img in enumerate(db_cluster.labels_.reshape(data_post_exp.shape[:3])):
    img[np.where(img != 0)] = 1
    im = axs[n].imshow(img)
    axs[n].axis('off')
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
    distances += abs(z31[:len(polarization_keys)] - z11[n])
    distances += np.linalg.norm(z33[len(polarization_keys):] - z13[n + xyz], axis=1)
    distances += abs(z31[len(polarization_keys):] - z11[n + xyz])

    nearest_neighbor_index = np.argmin(distances)

    min_indexes.append(nearest_neighbor_index)
    new_dat[n] = polarization_keys[nearest_neighbor_index]


colors = ['#394aff', '#b084ff','#9222ff']
cmap = LinearSegmentedColormap.from_list('mycmap', colors, N=256)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 10))
for n, img in enumerate(new_dat.reshape(data_post_exp[:5].shape[:3])):
    im = axs[n].imshow(img, cmap='bwr', aspect='auto', interpolation='bicubic')
    axs[n].axis('off')
    # if n == 9:
    #     fig.colorbar(im, ax=axs[n], orientation='vertical', fraction=.2)

cbar = fig.colorbar(im)
plt.show()

#%%
line = new_dat.reshape(data_post_exp[:5].shape[:3])
line = np.mean(line, axis=2)
for l in line:
    plt.plot(l)

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
