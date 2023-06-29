#%%
import hyperspy.api as hs
import os
path = r"/home/jinho93/project/GaN/3.0 at doped/EDS"

eds_3 = hs.load(os.path.join(path, 'Raw.emd'))
#%%
eds_p = hs.load(os.path.join(path.replace('3.0 at doped', 'Pristine'), 'Raw.emd'))

# %%
import matplotlib.pyplot as plt
import numpy as np
# plt.imshow(np.log(eds[-1].data[...,800:890].sum(axis=-1) + 1))
# %%
norm_factor = eds_3[-1].data.sum(axis=(0, 1))[1000] / eds_p[-1].data.sum(axis=(0, 1))[1000]
plt.semilogy(np.linspace(-0.2, 40.2, 4096), eds_3[-1].data.sum(axis=(0, 1)), c='b')
plt.semilogy(np.linspace(-0.2, 40.2, 4096), norm_factor * eds_p[-1].data.sum(axis=(0, 1)), c='r')
plt.xlim(1, 1.4)
plt.ylim(1e4, 1e6)
plt.plot([1.25361, 1.25361], [1e4, 1e5], c='k', linestyle='--')
# %%
plt.imshow(eds_3[-1].data[:200,:200, 1000:1100].sum(axis=-1))
# %%
