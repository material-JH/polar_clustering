#%%
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from abtem import *
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import PixelatedDetector
# atoms = read('E:/5.Script/2.ABTEM/1.ABTEM_practice/result/2021_09_17_abtem_practice/srtio3_100.cif')
from skimage.transform import resize


def crop_center(dp, crop_dim):
    nrx, nry, nkx, nky = dp.shape
    startx = nkx//2-(crop_dim[0]//2)
    starty = nky//2-(crop_dim[1]//2)
    return dp[:, :, starty:starty+crop_dim[0], startx:startx+crop_dim[0]]

def crop_center_image(dp, crop_dim):
    nky, nkx = dp.shape
    starty = nky//2-(crop_dim[0]//2)
    startx = nkx//2-(crop_dim[1]//2)
    return dp[starty:starty+crop_dim[0], startx:startx+crop_dim[0]]


class Stem:
    save_path = './processed_data/'

    def __init__(self, device) -> None:
        self.device = device

    def rotate_atom(self, tilt_angle, axis):
        self.atoms.rotate(tilt_angle, axis)
        
    def set_atom(self, file):
        self.atoms = read(file)
        
    def repeat_cell(self, rep):
        self.atoms *= rep
        
    def generate_pot(self, N, t):
        self.potential = Potential(self.atoms,  # or replace this with atoms
                      gpts=N,
                      slice_thickness= t,  # thickness: half u.c.  atoms.cell[2,2] / repz/1
                      parametrization='kirkland',
                    #   projection='infinite',
                      projection='infinite',
                      device=self.device).build(max_batch=32)

    def set_probe(self):
        self.probe = Probe(energy=300e3, semiangle_cutoff=2.16, defocus=0, focal_spread=20, device='cpu', rolloff=0.)
        self.probe.grid.match(self.potential)

    def set_scan(self, scan_width):

        self.gridscan = GridScan(start=[self.potential.extent[0]/2 - scan_width[0] / 2, 
                                        self.potential.extent[1]/2 - scan_width[0] / 2],
                    end=[self.potential.extent[0]/2 + scan_width[0] / 2, 
                         self.potential.extent[1]/2 + scan_width[0] / 2],
                    gpts=[1, 1])

        self.detector = PixelatedDetector(max_angle=None, resample=False)

    def scan(self, batch_size):
        return stem.probe.scan(self.gridscan, self.detector, self.potential, max_batch=batch_size, pbar=True)

num_Ti = 5
num_O = 3
num_Thikcness = 2
num_Tilt_x = 5
num_Tilt_y = 5
DP_y = 458
DP_x = 458
N = 512

lattice_constant = 3.94513
repeat_layer = 50

stem = Stem('cpu')
##########################################################################################################
## Ratio between the displacement of Ti and O
for i in range(5):
    for j in range(3):
        for k in range(3):
            for l in range(5):
                for m in range(5):

                    stem.set_atom('cif/BST_down_{}_{}.cif'.format(1.0*i,1.0*j))
## Thickness
                    thickness_layer = int(148 + 5*k)

                    stem.repeat_cell((repeat_layer, repeat_layer, thickness_layer))
##########################################################################################################
## Tilt
                    tilt_angle = -0.05 + 0.025 * l
                    tilt_angle_y =  -0.05 + 0.025 * m
                    stem.rotate_atom(tilt_angle, 'x')
                    stem.rotate_atom(tilt_angle_y,'y')
                    
                    stem.generate_pot(N, lattice_constant/2)
                    
                    stem.set_probe()
                    stem.set_scan((2 * 10, 2 * 10))
##########################################################################################################

                    measurement = stem.scan()

                    new_size = min(int(N * measurement.calibrations[2].sampling / measurement.calibrations[3].sampling),
                                   int(N * measurement.calibrations[3].sampling / measurement.calibrations[2].sampling))

                    test = np.zeros([1, 1, new_size, new_size])
                    for i in range(measurement.array.shape[0]):
                        for j in range(measurement.array.shape[1]):
                            if measurement.calibrations[2].sampling < measurement.calibrations[3].sampling:
                                test[i, j, :, :] = resize(measurement.array[i, j, :, :], [new_size, N])[:,
                                                   int((N - new_size) / 2):int((N - new_size) / 2) + new_size]
                            else:
                                test[i, j, :, :] = resize(measurement.array[i, j, :, :], [N, new_size])[
                                                   int((N - new_size) / 2):int((N - new_size) / 2) + new_size, :]

                    measurement_np = crop_center(test, [55, 55])

                    plt.imshow(measurement_np[0, 0, :, :])
                    plt.show()
                    array = measurement_np[0,0,:,:]
##########################################################################################################
## Correcting Distortion
                    array_r = resize(array, (np.shape(array)[0], np.shape(array)[1]*38/41))
                    array_r = crop_center_image(array_r, [50, 50])
                    plt.imshow(array_r)
                    plt.show()
#############################################################################################
## Collect the array
                    atoms = np.save('H:/5.Script/14.Machine_learning/1.Polarization/DP_array_{}_{}_{}_{}_{}.cif'.format(i * 0.5,0.5 * j,k,l,m),array_r)
                    # del array, probe, measurement_np, array_r, atoms, test,potential,new_size,measurement,scan_width,gridscan,detector
                    # import gc
                    # gc.collect(generation=2)
#############################################################################################
## Save the array
