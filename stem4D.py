import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from abtem import *
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import PixelatedDetector
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

def squaring(measurement, orig_size, new_size, N):
    
    tmp = np.zeros([orig_size[0], orig_size[1], new_size, new_size])
    for i in range(measurement.array.shape[0]):
        for j in range(measurement.array.shape[1]):
            if measurement.calibrations[2].sampling < measurement.calibrations[3].sampling:
                tmp[i, j] = resize(measurement.array[i, j], [new_size, N])[:, int((N - new_size) / 2):int((N - new_size) / 2) + new_size]
            else:
                tmp[i, j] = resize(measurement.array[i, j], [N, new_size])[int((N - new_size) / 2):int((N - new_size) / 2) + new_size, :]
    return tmp

class Stem:
    def __init__(self, device) -> None:
        self.device = device

    def rotate_atom(self, tilt_angle, axis):
        self.atoms.rotate(tilt_angle, axis)
        
    def set_atom_from_file(self, file_path):
        self.atoms = read(file_path)

    def set_atom(self, atoms):
        self.atoms = atoms
        
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
        self.probe = Probe(energy=300e3, semiangle_cutoff=2.16, defocus=0, focal_spread=20, device=self.device, rolloff=0.)
        self.probe.grid.match(self.potential)

    def set_scan(self, scan_width):

        self.gridscan = GridScan(start=[self.potential.extent[0]/2 - scan_width[0] / 2, 
                                        self.potential.extent[1]/2 - scan_width[0] / 2],
                    end=[self.potential.extent[0]/2 + scan_width[0] / 2, 
                         self.potential.extent[1]/2 + scan_width[0] / 2],
                    gpts=[1, 1])

        self.detector = PixelatedDetector(max_angle=None, resample=False)

    def scan(self, batch_size):
        return self.probe.scan(self.gridscan, self.detector, self.potential, max_batch=batch_size, pbar=False)
