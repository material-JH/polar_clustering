#%%
import hyperspy.api as hs

path = '/home/jinho93/project/NbSTO/20230616_ZPW_NbSTO'
# %%
import py4DSTEM

dataset = py4DSTEM.io.import_file(path + '/acquisition_6_good/scan_x128_y128.raw')

# %%
data = dataset.data
data[0,0][data[0,0] > 1e+7] = 1e+3
import matplotlib.pyplot as plt
plt.imshow(data[0,0])
#%%
import numpy as np
input_dim = data.shape[-2:]

imstack_train = data.reshape(-1, *input_dim)
cut = 37
imstack_train = imstack_train[:,cut:-cut, cut:-cut]
mean = np.mean(imstack_train)
std = np.std(imstack_train)
imstack_train = (imstack_train - mean) / std
ind_test = np.random.choice(range(len(imstack_train)), len(imstack_train) // 5, replace=False)
ind_train = np.setdiff1d(range(len(imstack_train)), ind_test)

#%%
from atomai.models import rVAE, VAE
from model.vae_model import *
import gc
gc.collect()
input_dim = imstack_train.shape[-2:]
#%%
rvae = regVAE(input_dim, latent_dim=4,
                        numlayers_encoder=2, numhidden_encoder=64,
                        numlayers_decoder=2, numhidden_decoder=64,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10)
#%%
plt.imshow(imstack_train[0])
#%%
rvae.fit(
    X_train= imstack_train[ind_train], 
    X_test = imstack_train[ind_test],
    training_cycles=50,
    batch_size=2 ** 8)

# %%

z_mean, z_log_var = rvae.encode(imstack_train)

# %%
imgs = z_mean.reshape(data.shape[0], data.shape[1], -1)
# %%

plt.imshow(imgs[..., [0, 1, 3]])
#%%

for i in range(2):
    alpha = imgs[..., i]
    alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    if i == 0:
        plt.imshow(imgs[..., i], alpha=alpha, cmap='Reds')
    else:
        plt.imshow(imgs[..., i], alpha=alpha, cmap='Blues')

    plt.xlim(0, 128 // 2)
    plt.ylim(0, 128 // 2)
    plt.axis('off')
# %%

eds = hs.load(path + '/EDS/STO0.5Nb SI HAADF 19.0 Mx 20230616 2003.emd')
# %%
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.log(eds[-1].data[...,800:890].sum(axis=-1) + 1))
# %%
plt.semilogy(eds[-1].data.sum(axis=(0, 1)))
plt.xlim(0, 1000)
# %%
eds[8].var()
# %%
eds[-1].plot()
#%%

plt.imshow(eds[-1].data[..., 230:245].sum(axis=-1))
# %%
