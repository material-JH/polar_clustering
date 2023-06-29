#%%
import hyperspy.api as hs
import os
path = r"/mnt/c/Users/USER/Downloads/"

eds_3 = hs.load(os.path.join(path, 'Spectrum-Image-40-x-800-x-1030_2020-03-04T071245.973786_40x800x1030_1 (aligned).dm4'))
eels = eds_3.data
# %%
eels2 = hs.load(os.path.join(path, 'Spectrum Image 184 x 562 x 1030 (aligned).dm3'))
# %%
eels2.shape
# %%
import matplotlib.pyplot as plt
# %%
plt.imshow(eels2.data.sum(axis=2))
# %%
import numpy as np
fig, ax = plt.subplots(1,2)
ax[0].imshow(eels.sum(axis=2).T)
ax[0].axis('off')
ax[1].imshow(np.log(eels + 1).sum(axis=0), aspect='auto')
# %%
plt.semilogy(np.linspace(-20, 200, eels.shape[-1]), eels.sum(axis=0)[0])
plt.ylim(1e3, 1e4)
plt.xlim(100, 180)
# %%
eels3 = hs.load(os.path.join(path, 'Spectrum Image 300 x 600 x 1030 (aligned).dm3'))
# %%
plt.imshow(eels[...,int(4.67 * 110)])
# %%
from atomai.models import jVAE
from atomai.utils import train_test_split
# %%
vae = jVAE((eels.shape[-1],), 10)
# %%
eels = (eels - eels.mean(axis=(0, 1))) / eels.std(axis=(0, 1))
train_set, test_set = train_test_split(eels.reshape(-1, eels.shape[-1]), test_size=0.1)

# %%
vae.fit(train_set, epochs=100, batch_size=256, X_test=test_set)
# %%
encoded_mean, encoded_log_var, z = vae.encode(eels.reshape(-1, eels.shape[-1]))

# %%
fig, ax = plt.subplots(2, 1)
ax[0].imshow(eels.sum(axis=-1), aspect='auto')
ax[1].imshow(encoded_mean.reshape(40, 800, -1)[..., 5], aspect='auto')
# %%
