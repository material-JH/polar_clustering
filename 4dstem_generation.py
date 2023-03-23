#%%
from stem4D import *

N = 512
lattice_constant = 3.94513
repeat_layer = 40
stem = Stem('cpu')
######################
name = 'POSCAR_REV'
#%%
def main(stem, suffix):
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
            np.save(f'output/DP_up{suffix}.npy', measurement_np)
        else:
            np.save(f'output/DP_dn{suffix}.npy', measurement_np)


#%%
thickness_layer = 27
for tilt_angle in np.linspace(-0.05, 0, 2):
    stem.set_atom_from_file(f'cif/{name}')
    cell = stem.atoms.cell
    stem.atoms.get_positions()
    stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
    stem.rotate_atom(tilt_angle, 'x')
    main(stem, tilt_angle)



#%%
for thickness_layer in range(20, 25):
    stem.set_atom_from_file(f'cif/{name}')
    cell = stem.atoms.cell
    stem.atoms.get_positions()
    stem.repeat_cell((round(repeat_layer * cell[1,1] / cell[0,0]), repeat_layer, thickness_layer))
    main(stem, thickness_layer)

    print(thickness_layer)
# %%
plt.imshow(measurement_np[0,0])
# %%
import glob
import os
for f in glob.glob('output/*'):
    if f.__contains__('DP'):
        os.remove(f)
