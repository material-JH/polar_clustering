from functools import reduce
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