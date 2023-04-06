#%%
import gc
import random
import cv2
from stem4D import *
from ase import Atoms
import copy
from main import *
from tqdm import tqdm

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


def select_atom(atoms_list):

    num_cell = 25
    pos = np.zeros((len(atoms_list),len(atoms_list[0]), 3))
    for i, atoms in enumerate(atoms_list):
        pos[i] = copy.deepcopy(atoms.get_positions()) - atoms.get_center_of_mass()

    emb = get_emb(pos.reshape(len(atoms_list), -1), min_dist=0.01)
    lbl = get_lbl(emb, num_cell)

    selected_atoms = []

    for i in range(num_cell):
        selected_atoms.append(atoms_list[random.choice(np.where(lbl == i)[0])])
    return selected_atoms

N = 512
lattice_constant = 3.94513
######################
repeat_layer = 20


stem = Stem('gpu')


for xdat_type in ['a', 'c', 'g']:
    selected_atoms = read(f'xdat/XDATCAR_{xdat_type}', index='::2')
    # selected_atoms = select_atom(atoms_list)
    for thickness_layer in tqdm(range(78, 83, 2), desc=f'{xdat_type} thickness :'):
        for tilt_angle in np.linspace(-0.10, 0, 3):
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
                    measurement_np = main(stem)
                    np.save(f'output/all/DP_{xdat_type}_{thickness_layer}_{round(tilt_angle, 4)}_{direction}_{n}_{polar}.npy', measurement_np)
                    gc.collect()
print('done')
# %%
import glob
import os
for f in glob.glob('output/*'):
    if f.__contains__('DP_g') or f.__contains__('DP_a_') or f.__contains__('DP_c_'):
        os.remove(f)
# %%
from tqdm import tqdm
import os

directory = 'output/all/'

# Iterate through each file in the directory
for filename in tqdm(os.listdir(directory)):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Replace 'old' with 'new' in the file name
        tmp = filename[:-4].split('_')
        tmp[5], tmp[6] = tmp[6], tmp[5]
        new_filename = '_'.join(tmp) + '.npy'
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
