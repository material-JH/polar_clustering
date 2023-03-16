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
import matplotlib
matplotlib.use('Qt5Agg')
import imutils as imutils

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

##
## npy set 2 = norm set 1
## npy set 1 - 데미지 있는 데이터

# Total data pic
p = 'D:/1.Experimental_data/7.삼성종기원과제_2022/2022_07_06_machine_learning_raw_data/npy_file/Ru/set2/bin_4/Ru_set2_0V_bin4.npy'
orig = np.load(p)
tem = np.sum(np.sum(orig, axis=1), axis=0)
fig = plt.figure(figsize=(16, 30))
ax = fig.add_subplot()
ax.imshow(tem)
#orig = np.rollaxis(np.rollaxis(orig, 2), 3, 1)

# read voltage data
data = np.zeros((5,38,10,100,50))

from PIL import Image
data_1 = []


for i in range(-2, 3):
    p_2 = 'H:/5.Script/14.Machine_learning/'
    im_1 = Image.open(p_2 + 'max_value_array_SRO_set4_%iV_re3.tif' % i)
    im_1 = np.array(im_1)

    path_3 = 'D:/1.Experimental_data/7.삼성종기원과제_2022/2022_08_20_SRO_4d/4D_data/set4/'
    image_name_3 = '{}_SRO_set4_crop_d.npy'.format(i)
    data_1_img = (np.load(path_3 + image_name_3))
    for m in range(38):
        for n in range(10):
            data_1_img[m,n,:,:] = data_1_img[m,n,:,:]/im_1[m,n]
    data_1.append(data_1_img)

data_1 = np.stack(data_1, axis=0)

data_2 = []
for i in range(-2, 3):
    p_3 = 'H:/5.Script/14.Machine_learning/'
    im_3 = Image.open(p_3 + 'max_value_array_SRO_set4_%iV_re3.tif' % i)
    im_3 = np.array(im_3)

    path_2 = 'D:/1.Experimental_data/7.삼성종기원과제_2022/2022_08_20_SRO_4d/4D_data/set4/'
    image_name_2 = '{}_SRO_set4_crop_u.npy'.format(i)
    data_2_img = (np.load(path_2 + image_name_2))
    for m in range(38):
        for n in range(10):
            data_2_img[m,n,:,:] = data_2_img[m, n, :, :] / im_3[m, n]
    data_2.append(data_2_img)

data_2 = np.stack(data_2, axis=0)

angle_1 = 180
for i in range(5):
    for j in range(38):
        for k in range(10):
            img_rot = imutils.rotate(data_2[i,j,k],angle_1)
            data_2[i,j,k] = img_rot


rows = 50
cols = 50
crow,ccol = (int)(rows/2),(int)(cols/2)
radius = 25
x,y = np.ogrid[:rows, :cols]
mask = np.sqrt((x - crow)**2 + (y-ccol)**2) <= radius
plt.figure()
plt.imshow(mask)


plt.figure()
plt.imshow(data[2,5,9,:,:])


bin_y = 5
bin_x = 10
#7,15,23
start_y = 20
start_x = 0

data_crop = np.zeros((50,50))
data_crop =+ data[:,start_y:start_y+bin_y,start_x+start_x:start_x+bin_x,:,:]
data_crop = np.sum(np.sum(data_crop,axis=2),axis=1)

for i in range(5):
    for j in range(38):
        for k in range(10):
            data_1_mask = data_1[i,j,k,:,:]*mask
            data_1[i,j,k,:,:] = data_1_mask

for i in range(5):
    for j in range(38):
        for k in range(10):
            data_2_mask = data_2[i,j,k,:,:]*mask
            data_2[i,j,k,:,:] = data_2_mask

data[:,:,:,0:50,0:50] = data_1
data[:,:,:,50:100,0:50] = data_2

plt.figure()
plt.imshow(data[2,15,5])


# plt.figure()
# plt.imshow(NormalizeData(data_crop[0]),vmax=1,vmin=0)
# plt.figure()
# plt.imshow(data_crop[4],vmax=1.1e6,vmin=1e4)

#data[0,10, 1, :, :] = data[0,10, 1, :, :]*0.6
# data[0,10, 2, :, :] = data[0,10, 2, :, :]*0.9
# data[0,10, 3, :, :] = data[0,10, 3, :, :]*0.9
# data[0,10, 4, :, :] = data[0,10, 4, :, :]*0.9
# data[0,10, 5, :, :] = data[0,10, 5, :, :]*0.9

# voltage = 0
# bin_y = 6
# bin_x = 9
# start_y = 13
# start_x = 0
#
# data_crop = np.zeros((50,50))
# for i in range(bin_y):
#     for j in range(bin_x):
#         data_crop =+ orig[start_y+i,start_x+j,:,:]
# data_crop = data_crop/(bin_y*bin_x)
# plt.figure()
# plt.imshow(data_crop)

#
# dp_vec2 = np.reshape(data, (ndata,x,y,int(kx*ky/2),2))
# plt.figure()
# plt.imshow(dp_vec2[2,15,5,:,:])


#
# for i in range(5):
#     for j in range(38):
#         for k in range(10):
#             data_1_crop = data[i,j,k,:,:]
#             data_1_crop = NormalizeData(data_1_crop)
#             data[i,j,k,:,:] = data_1_crop


ndata, x, y, kx, ky = np.shape(data)
dp_vec = np.reshape(data, (ndata * x * y, kx * ky))

fit = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=3,
    random_state=4
)
xy = fit.fit_transform(dp_vec)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xy[:,0],xy[:,1],xy[:,2])



labels, _ = one_round_clustering(2, xy)
labels = labels.reshape(5, 38, 10)
'''
fig = plt.figure(figsize=(45,30))
gs = fig.add_gridspec(1,5, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for i in range(5):
    axs[i].imshow(labels[i,:,:])
'''

labels[0:4,:5,:] = 1
labels[0:4,29:,:] = 1
labels[0:4,5:29,:] = 2

labels[4,:4,:] = 1
labels[4,28:,:] = 1
labels[4,4:28,:] = 2

#
# labels[0:4,:5,:] = 1
# labels[0:4,18:,:] = 1
# labels[0:4,5:18,:] = 2
#
# labels[4,:4,:] = 1
# labels[4,17:,:] = 1
# labels[4,4:17,:] = 2

# labels[0:4,:18,:] = 1
# labels[0:4,18:,:] = 1
# labels[0:4,18:29,:] = 2
#
# labels[4,:17,:] = 1
# labels[4,17:,:] = 1
# labels[4,17:28,:] = 2

# extract middle portion
val = 0
wow = None
for k, v in Counter(labels.reshape(-1)).items():
    if v > val:
        val = v
        wow = k

xybool = labels == 2
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
from scipy.spatial.distance import pdist, squareform

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
c_rgb = np.transpose(np.array(((xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0])),(xy0[:, 1] - np.min(xy0[:, 1])) / (np.max(xy0[:, 1]) - np.min(xy0[:, 1])) ,(xy0[:, 2] - np.min(xy0[:, 2])) / (np.max(xy0[:, 2]) - np.min(xy0[:, 2])))))
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=c_rgb )
colors_sub0 = [cm.viridis(v) for v in (xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0]))]
n = 0
for i, l in enumerate(labels_sub):
    if l == wow2:
        c[i] = colors_sub0[n]
        n += 1

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=c)



all_label = np.array([[[[0.0, 0.0, 0.0, 1.0] for _ in range(10)] for _ in range(38)] for _ in range(5)])
all_label[xybool, 0:3] = c_rgb
all_label = all_label.reshape(5, 38, 10, 4)

#
#
# color_number = 1230
#
# for i in range(5):
#     for j in range(38):
#         for k in range(10):
#             if all_label[i,j,k,0] == np.array(c[color_number][0]):
#                 if all_label[i, j, k, 1] == np.array(c[color_number][1]):
#                     if all_label[i, j, k, 2] == np.array(c[color_number][2]):
#                         if all_label[i, j, k, 3] == np.array(c[color_number][3]):
#                             array_color = [i,j,k]
#
# plt.figure()
# plt.imshow(data[array_color[0],array_color[1],array_color[2]])
#
#
# fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2], c=c[color_number])
# colors_sub0 = [cm.viridis(v) for v in (xy0[:, 0] - np.min(xy0[:, 0])) / (np.max(xy0[:, 0]) - np.min(xy0[:, 0]))]
# n = 0
# for i, l in enumerate(labels_sub):
#     if l == wow2:
#         c[i] = colors_sub0[n]
#         n += 1



fig = plt.figure(figsize=(45, 30))
gs = fig.add_gridspec(1, 5, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for i in range(5):
    axs[i].imshow(all_label[i, :, :])



# import cv2
#
#
# resize_img = cv2.resize(all_label[0, :, :], (0, 0), fx=10, fy=10, interpolation=cv2.INTER_AREA)
# cv2.imshow('$'.format(i),resize_img)
#



# line = np.mean(all_label, axis=2)
# std = np.std(all_label, axis=2,dtype = np.float32)
# line = np.moveaxis(np.mean(all_label, axis=2), 0, 1)
# std = np.moveaxis(np.std(all_label[:,:,:,0], axis=2), 0, 1)
# fig = plt.figure(figsize=(10, 30))
# ax = fig.subplots()
# ax.imshow(line)
# fig = plt.figure(figsize=(10, 30))
# ax = fig.subplots()
# ax.imshow(std)

# line_0 = line[:,:,:]
#
# from skimage import io
#
# line_0 = line_0.astype('float32')
# io.imsave("line_0_set_2_d_re.tif", line_0_ff)