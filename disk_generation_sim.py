#%%
import numpy as np
import os
from main import *
import matplotlib.pyplot as plt
from skimage.transform import resize

arr = []
name = []
n = 0
for file in os.listdir('output'):
    # if file.__contains__('DP_dn') or file.__contains__('DP_up'):
    if file.__contains__('DP_'):
        arr.append(np.load(f'output/{file}'))
        print(file)
        name.append(file)
        n += 1
for n in range(len(arr)):
    # arr[n] = resize(arr[n][0,0],  [r + 150 for r in arr[n][0,0].shape])
    arr[n] = resize(arr[n][0,0],  [r + 50 for r in arr[n][0,0].shape])
start_pos = [60, 158]
# start_pos = [12, 113]
# start_pos = [0, 0]
rad = 50

disk = []
for i in range(4):
    if i % 2 == 0:
        for n in range(len(arr)):
            arr[n] = arr[n][:, ::-1]
    for n in range(len(arr)):
        arr[n] = arr[n][::-1, :]
    for n in range(len(arr)):
        disk.append(arr[n][start_pos[0]:start_pos[0] + rad,
                        start_pos[1]:start_pos[1] + rad])
        if n % 20 == 0:
            plt.imshow(disk[n])
            plt.axis('off')
            plt.show()
# %%
np.save('output/disk_011_4.npy', np.array(disk))
print('saved')
# %%

fig, ax = plt.subplots(2, 5)
for i in range(5):
    for j in range(2):
        ax[j, i].imshow(disk[i + j * 5])
        ax[j, i].axis('off')
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
    ax.imshow(cv2.GaussianBlur(arr[new_val], (5, 5), 0), vmax=np.max(arr[new_val])/2, vmin=0)
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
