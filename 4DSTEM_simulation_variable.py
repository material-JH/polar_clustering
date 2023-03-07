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

##########################################################################################################
## Ratio between the displacement of Ti and O
for i in range(5):
    for j in range(3):
        for k in range(3):
            for l in range(5):
                for m in range(5):

                    import gc
                    import numpy as np
                    import matplotlib.pyplot as plt
                    from ase.io import read
                    from abtem import *
                    from abtem.detect import SegmentedDetector
                    from abtem.waves import Probe
                    from abtem.scan import GridScan
                    from abtem.detect import PixelatedDetector
                    from PIL import Image
                    import cupy as cp
                    import matplotlib

                    matplotlib.use('Qt5Agg')
                    # atoms = read('E:/5.Script/2.ABTEM/1.ABTEM_practice/result/2021_09_17_abtem_practice/srtio3_100.cif')
                    from skimage import io
                    from skimage.transform import resize

                    path = 'H:/5.Script/14.Machine_learning/1.Polarization/'

                    num_Ti = 5
                    num_O = 3
                    num_Thikcness = 2
                    num_Tilt_x = 5
                    num_Tilt_y = 5
                    DP_y = 458
                    DP_x = 458
                    N = 512
                    atoms = read('H:/5.Script/10.STEM_simulation/1.Polarization_mapping/3.cif/BST_down_{}_{}.cif'.format(1.0*i,1.0*j))

##########################################################################################################
## Thickness
                    repeat_layer = 50
                    thickness_layer = int(148 + 5*k)
                    lattice_constant = 3.94513
                    pos = np.array((0,0))
                    atoms *= (repeat_layer, repeat_layer, thickness_layer)

##########################################################################################################
## Tilt
                    tilt_angle = -0.05 + 0.025 * l
                    atoms.rotate(tilt_angle,'x')
                    tilt_angle_y =  -0.05 + 0.025 * m
                    atoms.rotate(tilt_angle_y,'y')

                    potential = Potential(atoms,
                                          gpts=N,
                                          projection='infinite',
                                          slice_thickness=  lattice_constant/2,    #lattice_constant/2,
                                          parametrization='kirkland', device='gpu').build(pbar=True)

##########################################################################################################

                    # pp = potential.project()
                    # potential.project().show();

##########################################################################################################
## Probe (a = 2.16 semi angle)
                    probe = Probe(energy=300e3, semiangle_cutoff=2.16, defocus=0, focal_spread=20, device='gpu', rolloff=0.)
                    probe.grid.match(potential)

                    scan_width = (2 * 10, 2 * 10)

                    gridscan = GridScan(start=[potential.extent[0] / 2 - scan_width[0] / 2,
                                               potential.extent[1] / 2 - scan_width[0] / 2],
                                        end=[potential.extent[0] / 2 + scan_width[0] / 2,
                                             potential.extent[1] / 2 + scan_width[0] / 2],
                                        gpts=[1, 1])

                    detector = PixelatedDetector(max_angle=None, resample=False)
                    measurement = probe.scan(gridscan, detector, potential, pbar=True)

                    import scipy.ndimage as ndimage
                    from skimage.transform import rescale, resize, downscale_local_mean

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

                    #fig1, ax1 = plt.subplots()
                    #potential.project().show(ax1, alpha=0.5)
                    gridscan.add_to_mpl_plot(ax1, facecolor='b')

                    measurement_np = crop_center(test, [55, 55])

                    #plt.figure(), plt.imshow(measurement_np[0, 0, :, :])
                    array = measurement_np[0,0,:,:]



##########################################################################################################
## Correcting Distortion
                    array_r = resize(array, (np.shape(array)[0], np.shape(array)[1]*38/41))
                    array_r = crop_center_image(array_r, [50, 50])
                    #plt.figure(), plt.imshow(array_r)

#############################################################################################
## Collect the array
                    atoms = np.save('H:/5.Script/14.Machine_learning/1.Polarization/DP_array_{}_{}_{}_{}_{}.cif'.format(i * 0.5,0.5 * j,k,l,m),array_r)
                    del array, probe, measurement_np, array_r, atoms, test,potential,new_size,measurement,scan_width,gridscan,detector
                    import gc
                    gc.collect(generation=2)
#############################################################################################
## Save the array
