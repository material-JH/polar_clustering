#%%
import gc
import random
import cv2
from stem4D import *
from ase import Atoms
import copy
from main import *

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
    
def main(stem: Stem):
    stem.generate_pot(N, lattice_constant/2)
    stem.set_probe(gaussian_spread=10, defocus=100)
    stem.set_scan((2, 2))
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

xdat_type = 'g'

atoms_list = read(f'xdat/XDATCAR_{xdat_type}', index=':')
stem = Stem('gpu')

num_cell = 25
pos = np.zeros((len(atoms_list),len(atoms_list[0]), 3))
for i, atoms in enumerate(atoms_list):
    pos[i] = copy.deepcopy(atoms.get_positions()) - atoms.get_center_of_mass()

emb = get_emb(pos.reshape(len(atoms_list), -1), min_dist=0.01)
lbl = get_lbl(emb, num_cell)

selected_atoms = []

for i in range(num_cell):
    selected_atoms.append(atoms_list[random.choice(np.where(lbl == i)[0])])

polar_arr = []
for atom in selected_atoms:
    polar_arr.append(get_polar(atom))
plt.plot(polar_arr)
plt.show()
# thickness_layer = 80
for thickness_layer in range(79, 82):
    for tilt_angle in np.linspace(-0.1, -0.075, 2):
        for direction in ['x', 'y']:
            for n, atoms in enumerate(selected_atoms):
                atoms = copy.deepcopy(atoms)
                atoms.cell = np.diag(np.diag(atoms.cell))
                stem.set_atom(atoms)
                stem.rotate_atom(90, 'x')   
                polar = round(get_polar(atoms))
                cell = stem.atoms.cell
                stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
                stem.rotate_atom(tilt_angle, direction)
                print(cell)
                measurement_np = main(stem)
                np.save(f'output/DP_{xdat_type}_{thickness_layer}_{round(tilt_angle, 4)}_{direction}_{polar}_{n}.npy', measurement_np)
                gc.collect()
print('done')
# %%
import glob
import os
for f in glob.glob('output/*'):
    if f.__contains__('DP_g') or f.__contains__('DP_a_') or f.__contains__('DP_c_'):
        os.remove(f)
# %%
