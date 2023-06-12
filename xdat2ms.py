#%%
import gc
from lib.stem4D import *
from ase import Atoms
from tqdm import tqdm
from ase.neighborlist import NeighborList

def get_polar(atoms:Atoms):
    apos = atoms.get_positions()
    polar = []
    cutoff_radius = 2.5
    nl = NeighborList([cutoff_radius / 2]*len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    # loop over each atom in the Atoms object and print its neighboring atoms
    for i in range(len(atoms)):
        if atoms[i].symbol == 'Ti':
        # get the indices of the neighboring atoms for the current atom
            neighbors, offsets = nl.get_neighbors(i)
            neighbors = [j for j in neighbors if atoms[j].symbol =='O']
            center = np.mean(apos[neighbors], axis=0)
            polar.append(apos[i] - center)
    return np.mean(polar, axis=0)

def get_polar2(atoms):
    a = atoms.cell[0, 0]
    layer_spacing = a / 4

    # loop over the atoms and identify the layer each belongs to
    layers = {}
    for atom in atoms:
        layer_num = int(atom.position[0] / layer_spacing + 0.5)
        if layer_num in layers:
            layers[layer_num].append(atom)
        else:
            layers[layer_num] = [atom]
    
    polar = []
    for layer_num, atoms in layers.items():
        if not 'Ti' in [atom.symbol for atom in atoms]:
            continue
        numTi = len([atom for atom in atoms if atom.symbol == 'Ti'])
        dip = 0
        for atom in atoms:
            if atom.symbol == 'Ti':
                dip += atom.position[0]
            elif atom.symbol == 'O':
                dip -= atom.position[0] / 2
        polar.append(dip / numTi)
    return np.mean(polar)
#%%
from sklearn.cluster import DBSCAN
from ase.io import read, write
import numpy as np
from ase import Atoms, Atom

def average_pos_atoms(atoms_list):
    pos = np.zeros((len(atoms_list),len(atoms_list[0]), 3))
    for i, atoms in enumerate(atoms_list):
        pos[i] = atoms.get_positions()

    tmp_atoms = atoms_list[0].copy()
    tmp_atoms.set_positions(np.mean(pos, axis=0))
    return tmp_atoms

selected_atoms = []

atoms_list = read(f'XDATCAR', index=':')
for i in range(0, len(atoms_list), 100):
    selected_atoms.append(average_pos_atoms(atoms_list[i:i+100]))

xy_atoms = []

elements = [['Ti'], ['O'], ['Sr', 'Ba']]

for i in range(0, len(selected_atoms)):
    ave_atoms = Atoms(cell=selected_atoms[i].cell, pbc=selected_atoms[i].pbc)
    for element in elements:
        tmp_atoms = selected_atoms[i][[atom.index for atom in selected_atoms[i] if atom.symbol in element]]
        xy_clusters = DBSCAN(min_samples=2).fit_predict(tmp_atoms.get_positions()[:, :2])
        xy_average = []
        for n in range(len(np.unique(xy_clusters))):
            xy_average.append(np.mean(tmp_atoms.get_positions()[xy_clusters == n], axis=0))
        for atom in xy_average:
            ave_atoms += Atom(element[0], position=atom)
        xy_atoms.append(ave_atoms)    

filename = 'XDATCAR_average'
for image in xy_atoms:
    write(filename, image, format='vasp', vasp5=True, direct=True, append=True)

#%%
from glob import glob
from tqdm import tqdm
from gpaw import GPAW

N = 2 ** 10
lattice_constant = 8.037805
######################

polars = []
repeat_layer = 16
Aatoms = []
# for n, atoms in tqdm(enumerate(selected_atoms)):
for n, finput in enumerate(glob('gpaw/random/*.gpw')):
    if n < 21:
        continue
    atoms = GPAW(finput, txt=None).atoms
    Aatoms.append(atoms)
    polar = round(get_polar2(atoms), 4)
    polars.append(polar)
    for thickness_layer in tqdm(range(78, 83, 2), desc=f'{n}_{polar}'):
        stem = Stem('gpu')
        stem.generate_pot_dft(finput, N // 2 ** 4, lattice_constant, (repeat_layer, repeat_layer, thickness_layer))
        # stem.set_atom(atoms)
        # stem.generate_pot(N // 2 ** 4, 3.91)
        # stem.potential = stem.potential.tile((repeat_layer,repeat_layer, thickness_layer))
        for gaussian in [0, 1, 10]:
            for tilt_angle in np.linspace(-0.10, 0.1, 5):
                for direction in ['x', 'y']:
                        foutput = f'/mnt/e/output/dft/DP_{n}_{polar}_{thickness_layer}_{gaussian}_{direction}_{round(tilt_angle, 4)}.npy'
                        if os.path.exists(foutput):
                            continue
                        if direction == 'x':
                            tilt = (tilt_angle * 10, 0)
                        else:
                            tilt = (0, tilt_angle * 10)

                        stem.set_probe(gaussian_spread=gaussian, defocus=0, tilt=tilt)
                        stem.set_scan((2, 2))
                        measurement = stem.scan(batch_size=32)
                        measurement.array = measurement.array.astype(np.float32)
                        new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                                        int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
                        test = squaring(measurement, [1,1], new_size, N)

                        measurement_np = crop_center(test, [55 * 4, 55 * 4])

                        np.save(foutput, measurement_np)
                        gc.collect()
print('done')
#%%
from ase.io import write
for atoms in Aatoms:
    write('XDATCAR_fix', atoms, format='vasp', vasp5=True, direct=True, append=True)
# %%
import glob
import os
for f in glob.glob('/mnt/e/output/dft/*'):
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
