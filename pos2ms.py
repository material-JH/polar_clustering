#%%
from copy import deepcopy
from stem4D import *
from glob import glob
import numpy as np
import imutils
import cv2

def get_polar(atom):
    anum = atom.get_atomic_numbers()
    apos = atom.get_scaled_positions()
    polar = 0
    pos_range = [[0.2, 0.3], [0.7, 0.8]]
    for num, pos in zip(anum, apos):
        for prange in pos_range:
            if pos[0] > prange[0] and pos[0] < prange[1]:
                if num == 22:
                    polar += pos[0] * 4
                elif num == 8:
                    polar += pos[0] * -2
                else:
                    polar += pos[0] * 2
    return polar

N = 2 ** 10
lattice_constant = 3.91
repeat_layer = 20
stem = Stem('gpu')
atoms_list = []
for f in sorted(glob('cif/*/CONTCAR')):
    atom = read(f)
    atoms_list.append(atom)
    atom = read(f)
    atom.rotate(180, 'z')
    atoms_list.append(atom)


refactor = repeat_layer / 20
# %%
test_set = False
for n, atoms in enumerate(atoms_list):
    if test_set:
        if not n % 40 == 0:
            continue
    for thickness_layer in range(78, 83, 4):
        inatoms = deepcopy(atoms)
        stem.set_atom(inatoms)
        polar = round(get_polar(inatoms), 5)
        cell = stem.atoms.cell
        stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
        stem.generate_pot(N, lattice_constant)
        stem.set_probe(defocus=1e+1, gaussian_spread=1e-5, tilt=(0,0), focal_spread=100)
        stem.set_scan((2, 2))

        measurement = stem.scan(batch_size=32)
        new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                        int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
        test = squaring(measurement, [1,1], new_size, N)
        new_size = int(new_size / refactor)
        # test = imutils.resize(test[0,0], width=new_size, height=new_size, inter=cv2.INTER_CUBIC)
        # test = np.expand_dims(test, axis=(0, 1))
        measurement_np = crop_center(test, [55 * 4, 55 * 4])
        if test_set:
            print(polar)
            plt.imshow(measurement_np[0,0], vmax=np.max(measurement_np[0,0]) / 3)
            plt.show()
        else:
            np.save(f'output/0k/DP_{thickness_layer}_{polar}.npy', measurement_np)

# %%
plt.imshow(measurement_np[0,0])
# %%
import glob
import os
for f in glob.glob('output/0k/*'):
    if f.__contains__('DP'):
        os.remove(f)
