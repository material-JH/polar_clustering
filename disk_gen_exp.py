#%%
import numpy as np
import matplotlib.pyplot as plt
from main import *
import cv2
import gc
#%%
data = load_data(r"/home/jinho93/project/BST/data/set3_Ru")
data = crop(data, 400, [70, 70])
gc.collect()

#%%
circle = get_circle_conv(42)

tmp = np.reshape(data, (-1, *data.shape[-2:]))
com = []
for i in tmp:
    com.append(get_center(i, circle))
    
#%%
data_post = fn_on_resized(data, rotate_by_cen, 82, com, list=True)
data_post = crop_from_center(data_post, 250, com, list=True)

#%%
plt.imshow(data_post[0,15,0])
plt.plot([125, 125], [0, 250], 'r')
plt.plot([0, 250], [125, 125], 'r')
#%%
disk_pos_002 = [9, 99]
disk_pos_011 = [55, 145]
data_post_002 = crop(data_post, 50, disk_pos_002)
data_post_011 = crop(data_post, 50, disk_pos_011)
data_post_002_norm = normalize_Data(data_post_002)
data_post_011_norm = normalize_Data(data_post_011)
n = 9
data_post_002_norm = fn_on_resized(data_post_002_norm, cv2.GaussianBlur, (n, n), 0)
data_post_011_norm = fn_on_resized(data_post_011_norm, cv2.GaussianBlur, (n, n), 0)

#%%
plt.imshow(data_post_002_norm[2,25,2])
#%%
np.save('output/set3_Ru_002.npy', data_post_002_norm)
# np.save('output/set2_SRO_011.npy', data_post_011_norm)
# %%
