import numpy as np
import matplotlib.pyplot as plt
from abtem import *
from abtem.waves import Probe   
from abtem.scan import GridScan
from abtem.detect import PixelatedDetector
from skimage.transform import resize

from gpaw import GPAW
from abtem.dft import GPAWPotential
from abtem.potentials import Potential
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
    
    def generate_pot_dft(self, path, N, latc, repeat):
        gpaw = GPAW(path, txt=None)
        dft_pot = GPAWPotential(gpaw, gpts=N, slice_thickness=latc / 4, storage=self.device)
        dft_array = dft_pot.build(max_batch=32)
        self.potential = dft_array.tile(repeat)

    def set_probe(self, energy=300e3, semiangle_cutoff=2.16, defocus=0, focal_spread=20, rolloff=0., gaussian_spread=0, tilt=(0, 0)):
        self.probe = Probe(device=self.device, rolloff=rolloff, gaussian_spread=gaussian_spread, energy=energy, semiangle_cutoff=semiangle_cutoff, defocus=defocus, focal_spread=focal_spread, tilt=tilt)
        self.probe.grid.match(self.potential)

    def set_scan_gpts(self, scan_width, gpts, angle=None):

        self.gridscan = GridScan(start=[self.potential.extent[0]/2 - scan_width[0] / 2, 
                                        self.potential.extent[1]/2 - scan_width[0] / 2],
                    end=[self.potential.extent[0]/2 + scan_width[1] / 2, 
                         self.potential.extent[1]/2 + scan_width[1] / 2],
                    gpts=gpts)

        self.detector = PixelatedDetector(max_angle=angle, resample=False)

    def set_scan_sampling(self, start, sampling_width, angle=None):

        self.gridscan = GridScan(start=start,
                    sampling=sampling_width)

        self.detector = PixelatedDetector(max_angle=angle, resample=False)

    def scan(self, batch_size):
        return self.probe.scan(self.gridscan, self.detector, self.potential, max_batch=batch_size, pbar=False)

from matplotlib.colors import LinearSegmentedColormap
def plot_dp(dp, vmax=1e-3, vmin=0):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # colors = ['#000000', 'blue', 'green', 'red', 'yellow', 'white']

    colors = ['#00000F', '#0000FF','#00FF00', '#FF0000', '#FFFF00', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors, N=256)
    # cmap = plt.cm.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    # Create a colormap object
    ax.imshow(dp, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.axis('off')
    plt.show()