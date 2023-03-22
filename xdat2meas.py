#%%
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
repeat_layer = 10
stem = Stem('cpu')
######################
name = 'POSCAR_REV'

thickness_layer = 10
# path = r'/home/jinho93/materials/oxides/perobskite/bsto/mp-1075943/'
path = r'/home/jinho93/materials/oxides/perobskite/bsto/mp-1075943/isif7/'
atoms_list = read(path + 'XDATCAR', index=':')

polar_arr = []
for atom in atoms_list:
    polar_arr.append(get_polar(atom))

plt.plot(polar_arr)
#%%
for n, atoms in enumerate(atoms_list[::20]):
    stem.set_atom(atoms)
    polar = round(get_polar(atoms))
    print(polar)
    cell = stem.atoms.cell
    stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
    stem.generate_pot(N, lattice_constant/2)
    stem.set_probe()
    stem.set_scan((2 * 10, 2 * 10))
    measurement = stem.scan(batch_size=16)
    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
    test = squaring(measurement, [1,1], new_size, N)
    measurement_np = crop_center(test, [55 * 4, 55 * 4])
    np.save(f'output/DP_{thickness_layer}_{n}_{polar}.npy', measurement_np)
    plt.imshow(measurement_np[0,0])
# %%
import glob
import os
for f in glob.glob('output/*'):
    print(f)
    if f.__contains__('DP'):
        os.remove(f)
# %%
