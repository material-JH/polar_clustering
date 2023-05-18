#%%
import copy
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi

#%%
data_post_exp = np.load('output/Fe2O3.npy')
#%%
data_post_exp = fn_on_resized(data_post_exp, normalize_Data)
#%%
imstack_all = data_post_exp.reshape(-1, data_post_exp.shape[-2], data_post_exp.shape[-1])

train_ind = np.random.choice(imstack_all.shape[0], imstack_all.shape[0] // 2, replace=False)
test_ind = np.setdiff1d(np.arange(imstack_all.shape[0]), train_ind)
imstack_test = imstack_all[test_ind]
imstack_train = imstack_all[train_ind]
#%%
plot_tk(imstack_train)
#%%
# Intitialize rVAE model
input_dim = imstack_train.shape[1:]
vae = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[2, 2],
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,
                        loss='mse', seed=1234)

if os.path.exists('output/jvae_fe2o3_norm.tar'):
    vae.load_weights('output/jvae_fe2o3_norm.tar')
    print('loaded weights')

#%%
vae.fit(
    X_train=imstack_train,
    X_test=imstack_test,
    training_cycles=300,
    batch_size=2 ** 8)

#%%
vae.save_weights('output/jvae_fe2o3_norm')
# rvae.save_weights('output/rvae_fe2o3_norm')
#%%
encoded_mean, encoded_sd, alpha = vae.encode(imstack_all)
#%%
plt.imshow(alpha[:,0].reshape((*data_post_exp.shape[:2], -1)))
# plt.colorbar()
#%%
decoded = []
for i in range(len(encoded_mean)):
    decoded.append(vae.decode(np.concatenate([encoded_mean[i], alpha[i]], axis=0))[0])
decoded = np.array(decoded)
#%%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
n = test_ind[2300]
axs[0].imshow(imstack_all[n])
axs[1].imshow(decoded[n])
plt.show()

#%%
plt.scatter(encoded_mean[:,0], encoded_mean[:,1], alpha=.1, color='red', label='exp')
#%%
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=data_post_exp.shape[:2])
img = encoded_mean.reshape((*data_post_exp.shape[:2], -1))
# img = np.concatenate((img, img), axis=-1)
img = img - img.min()
img = img / img.max()
img = 1 - img
img = np.concatenate([img, alpha[:,0].reshape((*data_post_exp.shape[:2], -1))], axis=-1)
axs.imshow(img)
axs.axis('off')
plt.show()
# %%
new = imstack_all - decoded
# %%
new_ind = alpha[:,0] > .5
# %%
vae2 = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[2],
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128, 
                        loss='ce', seed=1234)

#%%

vae2.load_weights('output/jvae_fe2o3_norm2.tar')

#%%
vae2.fit(
    X_train=imstack_all[new_ind],
    # X_test=imstack_test,
    training_cycles=300,
    batch_size=2 ** 8)
# %%
encoded_mean2, encoded_sd2, alpha2 = vae2.encode(imstack_all[new_ind])
#%%

abf = np.mean(np.load('output/Fe2O3.npy'), axis=(-1, -2))
abf = abf - abf.min()
abf = abf / abf.max()
#%%
new_alpha = np.zeros((len(encoded_mean), 2))
new_alpha[new_ind] = alpha2

img = alpha[:,0].reshape(data_post_exp.shape[:2])
img2 = alpha[:,2].reshape(data_post_exp.shape[:2])
img3 = new_alpha[:,0].reshape(data_post_exp.shape[:2])

norm = np.linalg.norm([img, img2, img3], axis=0) * .5

plt.imshow(np.stack([img, img2, img3, norm], axis=-1))

plt.imshow(abf, alpha=1-abf, cmap='gray')
#%%
new_ind2 = np.argwhere(new_alpha[:,0] > .5)[:,0]
#%%
n = 4
vae3 = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[2] * n,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128, 
                        loss='ce')
#%%
vae3.load_weights('output/jvae_fe2o3_norm3.tar')
vae3.manifold2d()
#%%
data_selected = np.load('output/Fe2O3.npy')
data_selected = data_selected.reshape((-1, *data_selected.shape[-2:]))
data_selected = data_selected[new_ind2]
data_selected -= np.mean(data_selected, axis=(1, 2), keepdims=True)
data_selected /= np.std(data_selected, axis=(1, 2), keepdims=True)
#%%
train_sel = np.random.choice(np.arange(len(data_selected)), size=int(len(data_selected) * .8), replace=False)
test_sel = np.setdiff1d(np.arange(len(data_selected)), train_sel)
#%%
vae3.fit(
    X_train=data_selected[train_sel],
    X_test=data_selected[test_sel],
    training_cycles=200,
    batch_size=2 ** 8)
#%%
encoded_mean3, encoded_sd3, alpha3 = vae3.encode(data_selected)
plt.plot(alpha3, alpha=.3)
#%%


#%%
new_ind3 = new_ind2[np.argwhere(alpha3[:,0] > .5)[:,0]]
new_ind4 = new_ind2[np.argwhere(alpha3[:,1] > .5)[:,0]]
ind_map = np.ones((len(encoded_mean), 4))
ind_map[new_ind3, 0] = 0
ind_map[new_ind4, 1] = 0
ind_map[:, 3] = np.linalg.norm(ind_map[:, :2], axis=-1) / 2
plt.imshow(ind_map.reshape((*data_post_exp.shape[:2], -1)))
plt.imshow(abf, alpha=1-abf, cmap='gray')
#%%
data_selected2 = np.load('output/Fe2O3.npy')
data_selected2 = data_selected2.reshape((-1, *data_selected2.shape[-2:]))
data_selected2 = data_selected2[new_ind3]
data_selected2 -= np.mean(data_selected2, axis=(1, 2), keepdims=True)
data_selected2 /= np.std(data_selected2, axis=(1, 2), keepdims=True)
#%%
n = 4
vae4 = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[2] * n,
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128, 
                        loss='ce')
#%%
train_sel = np.random.choice(np.arange(len(data_selected2)), size=int(len(data_selected2) * .8), replace=False)
test_sel = np.setdiff1d(new_ind3, train_sel)
#%%
vae4.fit(
    X_train=imstack_all[train_sel],
    X_test=imstack_all[test_sel],
    training_cycles=200,
    batch_size=2 ** 8)
# %%

vae.save_weights('output/jvae_fe2o3_norm')
vae2.save_weights('output/jvae_fe2o3_norm2')
vae3.save_weights('output/jvae_fe2o3_norm3')
# %%
