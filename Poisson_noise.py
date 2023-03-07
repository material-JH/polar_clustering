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
import hyperspy.api as hs
from photutils.datasets import make_noise_image
from skimage import io

path_3 = 'G:/5.Script/14.Machine_learning/1.Polarization/1.Simulation/1.2022_09_03_simulation/'
image_name_3 ='simulated_disk.dm4'
img = hs.load(path_3 + image_name_3).data

noise_mask = make_noise_image(np.shape(img), distribution='poisson', mean=1)

plt.figure()
plt.imshow(noise_mask)

noisy_img = img + noise_mask/4e5

plt.figure()
plt.imshow(noisy_img)

plt.figure()
plt.imshow(img)

path_3 = 'G:/5.Script/14.Machine_learning/1.Polarization/1.Simulation/1.2022_09_03_simulation/'
image_name_3 ='experimental_disk.dm4'
img_2 = hs.load(path_3 + image_name_3).data

img_2 = img_2 - 800

plt.figure()
plt.imshow(img_2)

img_norm = noisy_img/(np.sum(noisy_img))
img_2_norm = img_2/(np.sum(img_2))

path = 'G:/5.Script/14.Machine_learning/1.Polarization/'
a = img_norm.astype('float32')
io.imsave(path + "simulated_6.tif",a)
b = img_2_norm.astype('float32')
io.imsave(path + "experimental_6.tif",b)
