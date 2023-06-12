#%%
import gc
from lib.stem4D import *
from lib.plot import plot_tk

from ase import Atoms
from ase.io import read, write
from tqdm import tqdm
from ase.neighborlist import NeighborList

from glob import glob
from tqdm import tqdm
from gpaw import GPAW
import os
N = 2 ** 10
lattice_constant = 8.037805
######################

polars = []
repeat_layer = 5
Aatoms = []
selected_atoms = read('xdat/XDATCAR_average', index='33:')
cell_x = np.linalg.norm(selected_atoms[0].cell[0])
cell_y = np.linalg.norm(selected_atoms[0].cell[1])

#%%
import cv2
ncell = 2
for n, atoms in tqdm(enumerate(selected_atoms)):
    Aatoms.append(atoms)
    for thickness_layer in tqdm(range(77, 83, 2), desc=f'{n}'):
        stem = Stem('gpu')
        stem.set_atom(atoms)
        stem.generate_pot(N // 2 ** 2, lattice_constant / 2)
        print(stem.potential.extent)
        stem.potential = stem.potential.tile((repeat_layer,repeat_layer, thickness_layer))
        print(stem.potential.extent)
        for tilt_angle in np.linspace(-0.1, 0.1, 5):
            for direction in ['x', 'y'][::-1]:
                    foutput = f'/mnt/e/output/dft/DP_{n}_{thickness_layer}_{direction}_{round(tilt_angle, 4)}.npy'
                    if os.path.exists(foutput):
                        continue
                    if direction == 'x':
                        tilt = (tilt_angle * 10, 0)
                    else:
                        tilt = (0, tilt_angle * 1)

                    stem.set_probe(gaussian_spread=5, defocus=0, focal_spread=0, tilt=tilt)
                    stem.set_scan_gpts((cell_x, cell_y), (ncell, ncell))
                    measurement = stem.scan(batch_size=32)
                    measurement.array = measurement.array.astype(np.float32)
                    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
                    test = squaring(measurement, [ncell,ncell], new_size, N)

                    # measurement_np = cv2.rotate(measurement_np, cv2.ROTATE_90_CLOCKWISE)
                    # plt.imshow(measurement_np[0,0], vmax=np.max(measurement_np[0,0]) * 0.1)
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.xlabel('x')
                    # plt.ylabel('y')
                    measurement_np = crop_center(test, [55 * 4, 55 * 4])
                    plot_dp(measurement_np[0, 0], np.max(measurement_np[0,0]) / 3, 0)
                    raise
                    np.save(foutput, measurement_np)
                    gc.collect()
print('done')
# %%
plot_tk(measurement_np.reshape(-1, 55 * 4, 55 * 4))