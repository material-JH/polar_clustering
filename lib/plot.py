from functools import reduce
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_alpha(alpha):
    fig, ax = plt.subplots(1, 5, figsize=(15, 10))
    for i, img in enumerate(alpha.reshape(5, 38, 10)):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    
    return fig, ax

def plot_exp_vs_sim(rvae, zexp, zsim, aexp, asim, criterion, nmax=10):
    xyz = reduce((lambda x, y: x * y), zexp.shape[:3])
    # xyz //= 2
    tot = 0
    fig, ax = plt.subplots(1, 4, figsize=(5,2))
    for n in range(xyz):
        distances = np.linalg.norm(zsim - zexp[n], axis=1)
        # distances += np.abs(z31 - z11[n])
        # distances += np.linalg.norm(z33 - z13[n + xyz], axis=1)
        # distances += np.abs(z31 - z11[n + xyz])
        if np.min(distances) > criterion:
            continue
        tot += 1
        if tot > nmax:
            break
        nearest_neighbor_index = np.argmin(distances)
        # fig.suptitle(f'{n} {nearest_neighbor_index}')
        fig.suptitle('exp vs sim')
        ax[0].imshow(rvae.decode(np.array([*zexp[n], *aexp[n]]))[0])
        ax[0].axis('off')
        ax[1].imshow(rvae.decode(np.array([*zsim[nearest_neighbor_index], *asim[nearest_neighbor_index]]))[0])
        ax[1].axis('off')

    return fig, ax


def plot_tk(data, vmax=None, vmin=None):

    import tkinter

    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    # Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.figure import Figure

    import numpy as np

    root = tkinter.Tk()
    root.wm_title("Embedding in Tk")

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
    colors = ['#00000F', '#0000FF','#00FF00', '#FF0000', '#FFFF00', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors, N=256)

    ax.imshow(data[0], cmap=cmap)
    ax.axis('off')
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
        if vmax is None or vmin is None:
            ax.imshow(data[new_val], cmap=cmap)
        else:
            ax.imshow(data[new_val], cmap=cmap, vmax=vmax, vmin=vmin)
        ax.axis('off')
        # required to update canvas and attached toolbar!
        canvas.draw()

    slider_update = tkinter.Scale(root, from_=0, to=len(data), orient=tkinter.HORIZONTAL,
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
    plt.show(block=False)

def plot_random(data):
    fig, ax = plt.subplots(3, 3, figsize=(3, 3), dpi=100)
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(data[random.randint(0, len(data))])
            ax[i, j].axis('off')
    plt.show()