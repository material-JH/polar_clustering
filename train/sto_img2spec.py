#%%
import hyperspy.api as hs


path = '/home/jinho93/project/NbSTO/20230616_ZPW_NbSTO'
eds = hs.load(path + '/EDS/STO0.5Nb SI HAADF 19.0 Mx 20230616 2003.emd')
#%%

eds[7].save('haadf')
#%%

eds[-1].save('eds')
#%%

haadf = hs.load('haadf.hspy')
eds = hs.load('eds.hspy')
#%%
data_EDS = eds[-1].data
data_haadf = eds[7].data
# %%

print(data_EDS.data.shape)
print(data_haadf.data.shape)
# %%
import matplotlib.pyplot as plt

plt.imshow(data_EDS[...,40:].sum(axis=-1))
# %%
import numpy as np
plt.plot(np.log(data_EDS.sum(axis=(0,1)) + 1))
plt.xlim(40, 250)
# plt.ylim(0, 1e5)
# %%
import numpy as np
summed_haadf = data_haadf.data
summed_haadf = np.log(summed_haadf)
vmin = summed_haadf.min()
vmax = summed_haadf.max()
alpha = (summed_haadf - vmin) / (vmax - vmin)
#%%
plt.imshow(data_haadf[5:205, 5:205], cmap='gray')
plt.axis('off')
# %%
plt.imshow(data_haadf[-20:, -300:-10])
# %%
import atomai as aoi

cut = 200
imgs = []
for i in range(10, data_haadf.shape[0] - 10 - cut):
    for j in range(10, data_haadf.shape[1] - 10 - cut):
        imgs.append(data_haadf[i - 10:i+10, j - 10:j+10])

imgs = np.array(imgs)
#%%
import torch

spectra = []
for i in range(10, data_EDS.shape[0] - 10 - cut):
    for j in range(10, data_EDS.shape[1] - 10 - cut):
        # spectra.append(data_EDS[i, j, 1400:1700])
        spectra.append(data_EDS[i-1:i+1, j-1:j+2, 40:840].sum(axis=(0,1)))
#%%
spectra = np.array(spectra, dtype=np.int16)
#%%
spectra = torch.tensor(spectra, dtype=torch.int16)
#%%
plt.plot(spectra[-5])
# plt.plot(spectra.sum(axis=0))
plt.xlim(0, 1000)


#%%
imgs_train, spectra_train, imgs_test, spectra_test = aoi.utils.data_split(imgs, spectra, format_out="numpy")

in_dim = (20, 20)
out_dim = spectra.shape[-1]
#%%
import gc
gc.collect()
model2 = aoi.models.ImSpec(in_dim, out_dim, latent_dim=32, seed=2, 
                          batch_norm=True, encoder_downsampling=2, decoder_upsampling=True,
                          nbfilters_encoder=64, nbfilters_decoder=64, nblayers_encoder=2, nblayers_decoder=2,)

#%%
model2.fit(imgs_train, spectra_train, imgs_test, spectra_test,  # training data
          batch_size=128, print_loss=1,
          full_epoch=False, training_cycles=10, swa=False, batch_norm=True,
          plot_training_history=True)  # training parameters
# %%
model2.save_model('sto_img2spec3')
#%%
os.chdir('train')
model = aoi.models.load_model('sto_img2spec.tar')
#%%
import gc
gc.collect()
pred_spectra = model2.predict(imgs, norm=False, num_batches=128)
#%%
shape = int(np.sqrt(pred_spectra.shape[0]))
pred_eds = pred_spectra.reshape(shape, shape, -1)

#%%
import os
os.chdir('..')
from lib.plot import plot_tk
# %%
plot_tk(np.transpose(pred_eds, (2, 0, 1)))

# %%

np.save('pred_eds.npy', pred_eds)
# %%
