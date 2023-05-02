#%%
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from abtem.dft import GPAWPotential
from ase.io import read
from abtem import *
from stem4D import Stem
import os
from glob import glob
#%%
pots = []
for i in os.listdir('output/gpaw'):
    gpaw = GPAW('output/gpaw/' + i, txt=None)
    N = 2 ** 6
    dft_pot = GPAWPotential(gpaw, gpts=N, slice_thickness=latc / 4, storage='gpu')
    dft_array = dft_pot.build(max_batch=32)
    dft_potential = dft_array.tile((20,20, 80))
    pots.append(dft_potential)

# %%
from stem4D import squaring, crop_center
from glob import glob
from matplotlib.colors import LinearSegmentedColormap
colors = ['#00000F', '#0000FF','#00FF00', '#FF0000', '#FFFF00', '#FFFFFF']
# Create a custom color map
cmap = LinearSegmentedColormap.from_list('mycmap', colors, N=256)
stem = Stem(device='gpu')
# inatoms = read(sorted(glob('cif/*/CONTCAR'))[0])
N = 2 ** 10
latc = 7.906

for i in glob('output/gpaw/*.gpw'):
    stem.generate_pot_dft(i, N // 2 ** 4, latc, (20, 20, 80))
    stem.set_probe(defocus=0, gaussian_spread=0, tilt=(0,0), focal_spread=20)
    stem.set_scan((2, 2))
    measurement = stem.scan(batch_size=32)
    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
    test = squaring(measurement, [1,1], new_size, N)
    measurement_np = crop_center(test, [55 * 4, 55 * 4])
    plt.imshow(measurement_np[0,0], vmax=np.max(measurement_np[0,0]) * 0.3, cmap=cmap)
    plt.show()

# %%
