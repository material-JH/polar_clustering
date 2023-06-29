#%%
import os
from time import time
import numpy as np

import torch
import torch.nn as nn

from model.model import CNN1, FNN
from model.data import DataZ, DataZFlatten, get_loader

import matplotlib.pyplot as plt
import json
    
def use_model(data_loader, model, epoch):
    
    batch_time = AverageMeter()
    
    model.eval()
        
    t0 = time()
    outputs = []
    targets = []
    Bs = []
    idxs = []
    for i, j in enumerate(data_loader):
        idx, inputs = j
        idxs += idx
        # move input to cuda
        if next(model.parameters()).is_cuda:
            inputs = inputs.to(device='cuda')
        #compute output
        with torch.no_grad():
            output = model(inputs)
        outputs.extend(output.cpu().tolist())
        #measure elapsed time
        batch_time.update(time() - t0)
        t0 = time()
        
        s = 'Pred '
        
        print(s+': [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
          epoch, i, len(data_loader), batch_time=batch_time))
    
    print(s+' end: [{0}]\t'
      'Time {batch_time.sum:.3f}'.format(epoch, batch_time=batch_time))
    
    I = np.argsort(idxs)
    outputs = [outputs[i] for i in I]

    return np.array(outputs)
    #return outputs,targets,mpids,Bs
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    data_path_sim='output_pnu/z3.npz'
    data_path_exp='output_pnu/z1.npy'

    # Best Hyperparameters
    #var. for dataset loader
    batch_size = 512
    #var for training
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    # data = DataExp(data_path)

    norm_params = json.load(open('weights_pnu/norm_params.json','r'))

    
    data_exp = DataZFlatten(data_path_exp)
    data_exp.NormalizeWithParameters(*norm_params)
    data_sim = DataZFlatten(data_path_sim, 2)
    data_sim.NormalizeWithParameters(*norm_params)
    print('completed', time()-t,'sec')
    loader = get_loader(data_exp,batch_size=batch_size,idx_sets=[list(range(len(data_exp)))])[0]
    #build model
    # model = CNN1()
    model = FNN(data_exp.raw_Xs.shape[-1], h_dim=256)
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    os.makedirs('predict',exist_ok=True)
    outputs = []
    model.load_state_dict(torch.load('weights_pnu/W.pth.tar'))
    output = use_model(loader,model,0)
    # output = data_sim.revert_normalize(output)
    # output = data.revert_normalize(output)
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    # plt.plot(output)
    # json.dump([mpids,outputs,target,std],open('predict/Perov_All.json','w'))
output = data_sim.revert_normalize(output)
#%%
fig, ax = plt.subplots(1,4, figsize=(6, 5))
# vmax = np.max(output)
# vmin = np.min(output)
vmax = .15
vmin = -.15
# output[lbl==0] = 0
output_reshape = np.reshape(output, (5, 38, 10))
z1_10 = data_exp.raw_Xs[:,-8].reshape(5, 38, 10)
z1_0 = data_exp.raw_Xs[:,0].reshape(5, 38, 10)
ylim = np.array([34, 1])
for n, img in enumerate(output_reshape[::-1]):
    # zero_img = np.zeros_like(img)
    # zero_img[np.logical_and(-0.01 < img,  img < 0.01)] = 1
    # print(img.min(), img.max())
    if n % 2 == 1:
        continue
    n = n // 2

    xs_q, ys_q = np.meshgrid(np.linspace(0, 9, 10), np.linspace(0, 37, 38) )
    xs_q_grad = np.zeros_like(xs_q)
    ys_q_grad = img * -1
    # pcm = ax[n].imshow(img, cmap='RdBu', interpolation='bessel',vmin=vmin,vmax=vmax)
    # pcm = ax[n].imshow(img, cmap='RdBu_r',vmin=vmin,vmax=vmax, alpha=0.6, aspect='auto', interpolation='bicubic')
    ax[n].contourf(img, alpha=0.6, cmap='RdBu_r', vmin=vmin, vmax=vmax, levels=50)
    ax[n].quiver(xs_q, ys_q, xs_q_grad, ys_q_grad, width=0.02, scale=0.6, color='k', alpha=0.5)
    ax[n].set_ylim(*ylim)
    if n == 0:
        ylim -= 1
        ax[n].set_ylim(*ylim)
        ylim += 1
        # ax[n].set_ylim(-2, 38 - 2)
    # if n == 4:
    #     plt.colorbar(pcm, ax=ax[n], ticks=[vmin, 0, vmax])
    # ax[n].imshow(zero_img, cmap='gray_r')

    trueth = z1_0[n] < 0.5
    trueth = trueth.astype(int)
    trueth = trueth + (z1_10[n] > 0.5).astype(int)
    trueth = trueth.astype(bool)
    trueth = trueth.T
    # ax[n].imshow((trueth), cmap='gray_r', alpha=trueth * .5, aspect='auto')
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    # ax2 = ax[n].twiny()
    # ax2.plot(np.mean(img, axis=1) * -100, range(38), c='k')
    # ax2.set_xlim(vmin * 100, vmax * 100)
    # # ax2.tick_params(labelbottom=False)
    # ax2.plot(np.zeros(38), range(38), c='k', alpha=.5, linestyle='--')
    # ax2.set_xlabel(f'disp. (pm)\n\n{n - 2}V')
    x = range(38)
    if n == 0:
        x = range(1, 39)
        c = 'b'
    elif n == 1:
        c = 'gray'
    elif n == 2:
        c = 'r'
    ax[-1].plot(np.mean(img, axis=1) * -100, x, c=c, alpha=.5)

ax[-1].set_xlim(vmin * 100, vmax * 100)
ax[-1].set_ylim(*ylim)
# ax[-1].set_ylim(38, 0)
ax[-1].set_xticks([-10, 0, 10])
ax[-1].set_yticks([])
# ax[-1].axis('off')
# fig.colorbar(pcm, ax=ax[-1])
plt.show()
#%%
fig, ax = plt.subplots(1,4, figsize=(6, 5))
# vmax = np.max(output)
# vmin = np.min(output)
vmax = .15
vmin = -.15
# output[lbl==0] = 0
#%%
output_reshape = np.reshape(output, (5, 38, 10))
z1_10 = data_exp.raw_Xs[:,-8].reshape(5, 38, 10)
z1_0 = data_exp.raw_Xs[:,0].reshape(5, 38, 10)
ylim = np.array([34, 1])
for n, img in enumerate(output_reshape[::-1]):
    # zero_img = np.zeros_like(img)
    # zero_img[np.logical_and(-0.01 < img,  img < 0.01)] = 1
    # print(img.min(), img.max())
    if n % 2 == 1:
        continue
    n = n // 2

    xs_q, ys_q = np.meshgrid(np.linspace(0, 9, 10), np.linspace(0, 37, 38) )
    xs_q_grad = np.zeros_like(xs_q)
    ys_q_grad = img * -1
    # pcm = ax[n].imshow(img, cmap='RdBu', interpolation='bessel',vmin=vmin,vmax=vmax)
    # pcm = ax[n].imshow(img, cmap='RdBu_r',vmin=vmin,vmax=vmax, alpha=0.6, aspect='auto', interpolation='bicubic')
    ax[n].contourf(img, alpha=0.6, cmap='RdBu_r', vmin=vmin, vmax=vmax, levels=50)
    ax[n].quiver(xs_q, ys_q, xs_q_grad, ys_q_grad, width=0.02, scale=0.6, color='k', alpha=0.5)
    ax[n].set_ylim(*ylim)
    if n == 0:
        ylim -= 1
        ax[n].set_ylim(*ylim)
        ylim += 1
        # ax[n].set_ylim(-2, 38 - 2)
    # if n == 4:
    #     plt.colorbar(pcm, ax=ax[n], ticks=[vmin, 0, vmax])
    # ax[n].imshow(zero_img, cmap='gray_r')

    trueth = z1_0[n] < 0.5
    trueth = trueth.astype(int)
    trueth = trueth + (z1_10[n] > 0.5).astype(int)
    trueth = trueth.astype(bool)
    trueth = trueth.T
    # ax[n].imshow((trueth), cmap='gray_r', alpha=trueth * .5, aspect='auto')
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    # ax2 = ax[n].twiny()
    # ax2.plot(np.mean(img, axis=1) * -100, range(38), c='k')
    # ax2.set_xlim(vmin * 100, vmax * 100)
    # # ax2.tick_params(labelbottom=False)
    # ax2.plot(np.zeros(38), range(38), c='k', alpha=.5, linestyle='--')
    # ax2.set_xlabel(f'disp. (pm)\n\n{n - 2}V')
    x = range(38)
    if n == 0:
        x = range(1, 39)
        c = 'b'
    elif n == 1:
        c = 'gray'
    elif n == 2:
        c = 'r'
    ax[-1].plot(np.mean(img, axis=1) * -100, x, c=c, alpha=.5)

ax[-1].set_xlim(vmin * 100, vmax * 100)
ax[-1].set_ylim(*ylim)
# ax[-1].set_ylim(38, 0)
ax[-1].set_xticks([-10, 0, 10])
ax[-1].set_yticks([])
# ax[-1].axis('off')
# fig.colorbar(pcm, ax=ax[-1])
plt.show()
