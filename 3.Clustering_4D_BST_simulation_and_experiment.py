# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:19:49 2022

@author: SmartTouch 10th
"""

## 2022 07 15
## 이미지 normalization 필요


import numpy as np
import matplotlib.pyplot as plt
import umap
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from collections import Counter
from matplotlib import cm
from PIL import Image
import imutils as imutils
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from sklearn import preprocessing


def one_round_clustering(n_clusters, manifold_data):
    if np.shape(manifold_data)[1] > 1000:
        manifold_clustering_result = MiniBatchKMeans(n_clusters=n_clusters).fit(manifold_data)
    else:
        manifold_clustering_result = KMeans(n_clusters=n_clusters).fit(manifold_data)

    labels = manifold_clustering_result.labels_ + 1

    return labels, manifold_clustering_result.cluster_centers_


def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
            rcos * np.eye(3) +
            rsin * np.array([
        [0, -w, v],
        [w, 0, -u],
        [-v, u, 0]
    ]) +
            (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Total data pic
# p = 'D:/1.Experimental_data/7.삼성종기원과제_2022/2022_07_06_machine_learning_raw_data/npy_file/Ru/set1/bin_4/Ru_set1_0V_bin4.npy'
# orig = np.load(p)
# tem = np.sum(np.sum(orig, axis=1), axis=0)
# fig = plt.figure(figsize=(16, 30))
# ax = fig.add_subplot()
# ax.imshow(tem)
#orig = np.rollaxis(np.rollaxis(orig, 2), 3, 1)

# read voltage data
data = []
for i in range(-2, 3):
    path_3 = 'D:/1.Experimental_data/7.삼성종기원과제_2022/2022_08_20_SRO_4d/4D_data/set4/'
    image_name_3 = '{}_SRO_set4_crop_d.npy'.format(i)
    data.append(np.load(path_3 + image_name_3))
data = np.stack(data, axis=0)


# read simulated data
data_2 = []
for i in range(0,31):
    displacement_number = int(i*0.1)
    path_3 = 'H:/5.Script/10.STEM_simulation/1.Polarization_mapping/2.result/3.CBED_thickness_check/'
    image_name_3 ='600a_{}.tif'.format(displacement_number,i)
    data_2.append(np.array(Image.open(path_3+image_name_3)))
data_2 = np.stack(data_2, axis=0)
data_reshape = np.reshape(data_2,(31,1,1283,1283))


#
# ######################################################################################
# ## 1. Normalization (전체 intensity로 나눠주기) - 총 intensity 합은 언제나 같아야 함
#
# image_pixel_norm = np.zeros((1283,1283))
# data_reshape_norm = np.zeros((31,1,1283,1283))
#
# for i in range(31):
# 	for j in range(1):
# 		image_pixel_norm[:][:] = data_reshape[i][j][:][:]
#         image_pixel_norm = image_pixel_norm/(np.sum(image_pixel_norm))
#         data_reshape_norm[i][j][:][:] = image_pixel_norm


######################################################################################
## 2. Cropping the image

angle_1 = 90
angle_2 = -90
mask = 150

img_pixel = np.zeros((mask,mask))
data_rotate= np.zeros((31,2,mask,mask))

first_mask_coordi = (569,244)
second_mask_coordi = (569,893)

lu, ru, tu, bu = [first_mask_coordi[0], first_mask_coordi[0]+mask, first_mask_coordi[1], first_mask_coordi[1]+mask]  # 0-20 disc position
#lu, ru, tu, bu = [second_mask_coordi[0], second_mask_coordi[0]+mask, second_mask_coordi[1], second_mask_coordi[1]+mask]  # 020 disc position
data_crop= data_reshape[:, :, lu:ru, tu:bu]

for i in range(31):
    img_pixel[:][:] = data_crop[i][0][:][:]
    img_rot = imutils.rotate(img_pixel,angle_1)
    data_rotate[i][0] = img_rot

lu, ru, tu, bu = [second_mask_coordi[0], second_mask_coordi[0]+mask, second_mask_coordi[1], second_mask_coordi[1]+mask]  # 020 disc position
#lu, ru, tu, bu = [second_mask_coordi[0], second_mask_coordi[0]+mask, second_mask_coordi[1], second_mask_coordi[1]+mask]  # 020 disc position
data_crop= data_reshape[:, :, lu:ru, tu:bu]

for i in range(31):
    img_pixel[:][:] = data_crop[i][0][:][:]
    img_rot = imutils.rotate(img_pixel,angle_2)
    data_rotate[i][1] = img_rot

data_rotate_rescale = np.zeros((31,2,50,50))
for i in range(31):
    for j in range(2):
        img_a = data_rotate[i][j]
        img_a = resize(img_a,(50,50))
        data_rotate_rescale[i][j] = img_a

data_2 = data_rotate_rescale

for i in range(5):
    for j in range(38):
        for k in range(10):
            data_1_crop = data[i,j,k,:,:]
            data_1_crop = NormalizeData(data_1_crop)
            data[i,j,k,:,:] = data_1_crop

for i in range(31):
    for j in range(2):
        data_2_crop = data_2[i, j, :, :]
        data_2_crop = NormalizeData(data_2_crop)
        data_2[i, j, :, :] = data_2_crop

plt.figure()
plt.imshow(data[0,10,5],vmin = 0 , vmax = 1)
plt.figure()
plt.imshow(data_2[15,1],vmin = 0 , vmax = 1)
# plt.figure()
# plt.imshow(data[1,0])
# plt.figure()
# plt.imshow(data[2,0])
# plt.figure()
# plt.imshow(data[3,0])
# plt.figure()
# plt.imshow(data[4,0])
# plt.figure()
# plt.imshow(data[5,0])
# plt.figure()
# plt.imshow(data[6,0])
# plt.figure()
# plt.imshow(data[7,0])
# plt.figure()
# plt.imshow(data[8,0])



ndata, x, y, kx, ky = np.shape(data)
dp_vec_1 = np.reshape(data, (ndata * x * y, kx * ky))

x_s, y_s, kx_s, ky_s = np.shape(data_2)
dp_vec_2 = np.reshape(data_2, (x_s * y_s, kx_s * ky_s))

dp_vec = np.zeros((np.shape(dp_vec_1)[0]+np.shape(dp_vec_2)[0], np.shape(dp_vec_1)[1]))
dp_vec[:np.shape(dp_vec_1)[0]] = dp_vec_1
dp_vec[np.shape(dp_vec_1)[0]:np.shape(dp_vec_1)[0]+np.shape(dp_vec_2)[0]] = dp_vec_2

fit = umap.UMAP(
    n_neighbors=80,
    min_dist=0.3,
    n_components=3,
    random_state=4
)
xy = fit.fit_transform(dp_vec)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xy[:,0],xy[:,1],xy[:,2])

labels, _ = one_round_clustering(3, xy)

labels_1 = labels[0:1900]
labels_2 = labels[1900:1962]
labels_1 = labels_1.reshape(5, 38, 10)
labels_2 = labels_2.reshape(31, 2)

plt.figure()
plt.imshow(labels_1[0])
plt.figure()
plt.imshow(labels_2)


xy0 = xy
labels_sub =labels

i, j = np.unravel_index(np.argmax(squareform(pdist(xy0))), (xy0.shape[0], xy0.shape[0]))
p1 = xy0[i, :]
p2 = xy0[j, :]
avg = np.mean([p1, p2], axis=0)
xy0 -= avg
p1 = xy0[i, :]
p2 = xy0[j, :]
rot_mat = get_rotation_matrix(p2 - p1)
xy0 = np.matmul(rot_mat, xy0.T).T
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=(xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0])))
colors_sub0 = [cm.viridis(v) for v in (xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0]))]
n = 0
wow2=1

m = SpectralClustering(1)
labels_sub = m.fit_predict(xy)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
c = []
lmax = np.max(labels_sub)

for l in labels_sub:
    c.append(cm.Set1(l / lmax))
ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2],  s=100)

for i, l in enumerate(labels_sub):
    c[i] = colors_sub0[n]
    n += 1

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=c)


all_label = np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(1962)])
for i in range(1962):
    all_label[i,:] = colors_sub0[i]

all_label_1 = all_label[:1900,:]
all_label_2 = all_label[1900:1962,:]
all_label_1 = all_label_1.reshape(5,38,10,4)
all_label_2 = all_label_2.reshape(31,2,4)

fig = plt.figure(figsize=(15, 30))
gs = fig.add_gridspec(1, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
axs.imshow(all_label_1[0, :, :])

fig = plt.figure(figsize=(15, 30))
gs = fig.add_gridspec(1, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
axs.imshow(all_label_2[ :, :])

'''
fig = plt.figure(figsize=(45,30))
gs = fig.add_gridspec(1,5, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for i in range(5):
    axs[i].imshow(labels[i,:,:])
'''
# extract middle portion (label 1)
wow = int(1)
xybool = labels == wow
dp_vec_sub = dp_vec[xybool.reshape(-1), :]

fit = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=3,
    random_state=4
)
xy = fit.fit_transform(dp_vec_sub)

m = SpectralClustering(1)
labels_sub = m.fit_predict(xy)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
c = []
lmax = np.max(labels_sub)
for l in labels_sub:
    c.append(cm.Set1(l / lmax))
ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2],  s=100)

val = 0
wow2 = None
for k, v in Counter(labels_sub.reshape(-1)).items():
    if v > val:
        val = v
        wow2 = k

xy0 = xy[labels_sub == wow2]


i, j = np.unravel_index(np.argmax(squareform(pdist(xy0))), (xy0.shape[0], xy0.shape[0]))
p1 = xy0[i, :]
p2 = xy0[j, :]
avg = np.mean([p1, p2], axis=0)
xy0 -= avg
p1 = xy0[i, :]
p2 = xy0[j, :]
rot_mat = get_rotation_matrix(p2 - p1)
xy0 = np.matmul(rot_mat, xy0.T).T
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=(xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0])))
colors_sub0 = [cm.gray(v) for v in (xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0]))]
n = 0
for i, l in enumerate(labels_sub):
    if l == wow2:
        c[i] = colors_sub0[n]
        n += 1


fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=c)


all_label = np.array([[[0.0, 0.0, 0.0, 0.0] for _ in range(2)] for _ in range(31)])
all_label[xybool, :] = c
all_label = all_label.reshape(31,2,4)


fig = plt.figure(figsize=(15, 30))
gs = fig.add_gridspec(1, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
axs.imshow(all_label[ :, :])

line = np.mean(all_label, axis=2)
line = np.moveaxis(np.mean(all_label, axis=2), 0, 1)
fig = plt.figure(figsize=(10, 30))
ax = fig.subplots()
ax.imshow(line)


# line_0 = line[:,:,:]
#
# from skimage import io
#
# line_0 = line_0.astype('float32')
# io.imsave("line_0_set_2_d_re.tif", line_0_ff)