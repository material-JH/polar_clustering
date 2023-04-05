#%%
import random
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

#%%

arr = []
fnames = []
n = 0

circle = get_circle_conv(45)
for file in os.listdir('output'):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP_a') or file.__contains__('DP_c') or file.__contains__('DP_g'):
        arr.append(np.load(f'output/{file}'))
        fnames.append(file)
        n += 1
for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
    # arr[n] = arr[n]np.roll(arr[n], get_center(), axis=0)
start_pos_011 = [60, 158]
start_pos_002 = [12, 113]
start_pos_m002 = [212, 113]
arr = np.array(arr)
rad = 54
#%%
sep = {}
for n, i in enumerate(set([i.split('_')[3] for i in fnames])):
    sep[i] = n

disk = {}
for i in sep.keys():
    disk[i] = []
    
for i in range(4):
    arr = arr[::-1,:]
    if i % 2 == 0:
        arr = arr[:, ::-1]
    if i // 2 == 0:
        start_pos_011[1] = 159
    else:
        start_pos_011[1] = 156
    # for n in range(1):
    for n in range(len(arr)):
        dn = np.sum(crop(arr[n], rad, start_pos_002))
        up = np.sum(crop(arr[n], rad, start_pos_m002))
        
        img = crop(arr[n], rad, start_pos_011)
        name = fnames[n]
        disk[name.split('_')[3]].append(img)
    # prime = get_center(disk[0], circle)
    # for i in range(len(disk)):
    #     disk[i] = np.roll(disk[i], prime[0] - get_center(disk[i], circle)[0], axis=0)
    #     disk[i] = np.roll(disk[i], prime[1] - get_center(disk[i], circle)[1], axis=1)
    #     print(get_center(disk[i], circle))
    # print(prime)

for i in sep.keys():
    disk[i] = np.array(disk[i])
    print(len(disk[i]))

# %%
np.savez('output/disk_011_5.npz', **disk)
print('saved')

#%%
for i in random.sample(range(len(disk_up)), 5):
    plt.imshow(disk_up[i])
    plt.show()
#%%

import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
import cv2

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
t = np.arange(0, 3, .01)

ax.imshow(arr[0])

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)

def update_frequency(new_val):
    new_val = int(new_val)
    ax.clear()
    # ax.imshow(imutils.rotate(data[0, 20, 0], int(new_val), com[200]))
    # ax.imshow(imutils.rotate(data.sum(axis=2)[2, new_val], int(82), com[200]))
    ax.imshow(cv2.GaussianBlur(arr[new_val], (5, 5), 0), vmax=np.max(arr[new_val])/8, vmin=0)
    ax.axis('off')
    # required to update canvas and attached toolbar!
    canvas.draw()

slider_update = tkinter.Scale(root, from_=0, to=len(arr), orient=tkinter.HORIZONTAL,
                              command=update_frequency, label="Frequency [Hz]")

# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
 
tkinter.mainloop()
# %%
plt.imshow(arr[220])
# %%

