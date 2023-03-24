#%%
import cv2
from stem4D import *
from ase import Atoms
def get_polar(atom:Atoms):
    anum = atom.get_atomic_numbers()
    apos = atom.get_positions()
    polar = 0
    for num, pos in zip(anum, apos):
        if num == 22:
            polar += pos[2] * 4
        elif num == 8:
            polar += pos[2] * -2
        else:
            polar += pos[2] * 2
    
    return polar
    

N = 512
lattice_constant = 3.94513
stem = Stem('gpu')
######################
# repeat_layer = 20
# thickness_layer = 100
# atoms_list = read('xdat/XDATCAR', index=':')
repeat_layer = 5
thickness_layer = 23
atoms_list = read('xdat/XDATCAR_large', index=':')

polar_arr = []
for atom in atoms_list:
    polar_arr.append(get_polar(atom))

for n, atoms in enumerate(atoms_list[::50]):
    stem.set_atom(atoms)
    polar = round(get_polar(atoms))
    cell = stem.atoms.cell
    stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
    print(cell)
    stem.generate_pot(N, lattice_constant/2)
    stem.set_probe()
    stem.set_scan((2 * 10, 2 * 10))
    measurement = stem.scan(batch_size=16)
    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
    test = squaring(measurement, [1,1], new_size, N)
    measurement_np = crop_center(test, [55 * 4, 55 * 4])
    np.save(f'output/DP_{thickness_layer}_{polar}.npy', measurement_np)
    if n % 2 == 0:
        n = 7
        blur = cv2.GaussianBlur(measurement_np[0,0], (n, n), 0)

        plt.imshow(blur,vmax=np.max(blur) / 2)
        plt.show()
# %%
import glob
import os
for f in glob.glob('output/*'):
    if f.__contains__('DP_22'):
        os.remove(f)
# %%
