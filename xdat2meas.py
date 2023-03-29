#%%
import gc
import cv2
from stem4D import *
from ase import Atoms
import copy

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
    
def main(stem):
    stem.generate_pot(N, lattice_constant/2)
    stem.set_probe()
    stem.set_scan((2 * 10, 2 * 10))
    measurement = stem.scan(batch_size=32)
    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
    test = squaring(measurement, [1,1], new_size, N)
    measurement_np = crop_center(test, [55 * 4, 55 * 4])
    return measurement_np

N = 512
lattice_constant = 3.94513
######################
repeat_layer = 20
atoms_list = read('xdat/XDATCAR', index=':')
stem = Stem('gpu')
# repeat_layer = 5
# thickness_layer = 23
# repeat_layer = 10
# thickness_layer = 78
# atoms_list = read('xdat/XDATCAR_strain2', index=':')

polar_arr = []
for atom in atoms_list:
    polar_arr.append(get_polar(atom))
plt.plot(polar_arr)
plt.show()

for thickness_layer in range(78, 83):
    for tilt_angle in np.linspace(-0.05, 0.05, 5):
        for direction in ['x', 'y']:
            for n, atoms in enumerate(atoms_list[::20]):
                atoms = copy.deepcopy(atoms)
                stem.set_atom(atoms)
                polar = round(get_polar(atoms))
                cell = stem.atoms.cell
                stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
                stem.rotate_atom(tilt_angle, direction)
                print(cell)
                measurement_np = main(stem)
                np.save(f'output/DP_{thickness_layer}_{round(tilt_angle, 4)}_{direction}_{polar}_{n}.npy', measurement_np)
                gc.collect()
print('done')
# %%
import glob
import os
for f in glob.glob('output/*'):
    if f.__contains__('DP_'):
        os.remove(f)
# %%
