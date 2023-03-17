#%%
from stem4D import *
from pymatgen.core import Structure

N = 512
lattice_constant = 3.94513
repeat_layer = 40
thickness_layer = 15
stem = Stem('cpu')
######################
name = 'POSCAR_REV'
stem.set_atom(f'cif/{name}')
cell = stem.atoms.cell
stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))

for i in range(2):
    stem.rotate_atom(180, 'z')
    stem.generate_pot(N, lattice_constant/2)
    stem.set_probe()
    stem.set_scan((2 * 10, 2 * 10))
    measurement = stem.scan(batch_size=16)
    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                    int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))
    test = squaring(measurement, [1,1], new_size, N)
    measurement_np = crop_center(test, [55 * 4, 55 * 4])
    if i == 0:
        np.save('output/DP_up.npy', measurement_np)
    else:
        np.save('output/DP_dn.npy', measurement_np)
# %%
