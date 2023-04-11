#%%
import numpy as np
import matplotlib.pyplot as plt
from main import *
import cv2

data = load_data(r"/home/jinho93/project/BST/data/set4_Ru", contained='0.dm')
#%%
circle = get_circle_conv(40)

tmp = np.reshape(data, (-1, *data.shape[-2:]))
com = []
for i in tmp:
    com.append(get_center(i, circle))
    
#%%
data_post = fn_on_resized(data, rotate_by_cen, 82, com, list=True)
data_post = crop_from_center(data_post, 250, com, list=True)
data_post = normalize_Data(data_post)

#%%
plt.imshow(data_post[0,10,0])

#%%
disk_pos_002 = [7, 100]
disk_pos_011 = [55, 145]
data_post_002 = crop(data_post, 50, disk_pos_002, list=True)
data_post_011 = crop(data_post, 50, disk_pos_011, list=True)
data_post_002_norm = normalize_Data(data_post_002)
data_post_011_norm = normalize_Data(data_post_011)
n = 9
data_post_002_norm = fn_on_resized(data_post_002_norm, cv2.GaussianBlur, (n, n), 0)
data_post_011_norm = fn_on_resized(data_post_011_norm, cv2.GaussianBlur, (n, n), 0)

#%%
from scipy import ndimage
img = data_post_011_norm[0,10,0]

def sobel(img):

    sobel_x = ndimage.sobel(img, axis=0)
    sobel_y = ndimage.sobel(img, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    return edges
#%%
plt.imshow(crop_from_center(data_post_011_norm, 42, com)[0,10,0])
#%%

new_size = 42
circle = get_circle_conv(new_size)

com = fn_on_resized(fn_on_resized(data_post_011_norm, sobel), get_center, circle)
com = np.reshape(com, (-1, *com.shape[-1:]))
com = [list(map(int, c[::-1])) for c in com]

for n, c in enumerate(com):
    if c[0] < new_size // 2 or c[0] > 55 - new_size // 2 or c[1] < new_size // 2 or c[1] > 55 - new_size // 2:
        com[n] = [27, 27]
data_post_011_norm = crop_from_center(data_post_011_norm, new_size, com)

com = fn_on_resized(data_post_002_norm, get_center, circle)
com = np.reshape(com, (-1, *com.shape[-1:]))
com = [list(map(int, c[::-1])) for c in com]

for n, c in enumerate(com):
    if c[0] < new_size // 2 or c[0] > 55 - new_size // 2 or c[1] < new_size // 2 or c[1] > 55 - new_size // 2:
        com[n] = [27, 27]

data_post_002_norm = crop_from_center(data_post_002_norm, new_size, com)
#%%

# apply high limit to 1
data_post[data_post > 2.] = 2.
plot_tk(data_post.reshape(-1, data_post.shape[-2], data_post.shape[-1]))
# %%
