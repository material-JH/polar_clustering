#%%
import os
import json
from time import time
from sklearn import metrics
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from model.model import NN
from model.data import DataExp, get_loader, Data

from glob import glob
import matplotlib.pyplot as plt

    
def use_model(data_loader, model, epoch):
    
    batch_time = AverageMeter()
    
    model.eval()
        
    t0 = time()
    outputs = []
    targets = []
    Bs = []
    for i, j in enumerate(data_loader):
        idx, inputs = j
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
    data_path_exp= './output/set1_SRO_002.npy'
    data_path_sim= './output/disk_002_dft.npz'
    # Best Hyperparameters
    atom_fea_len = 64
    n_conv = 1
    lr_decay_rate = 0.99
    #var. for dataset loader
    batch_size = 512
    #var for training
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    # data = DataExp(data_path)
    data_exp = DataExp(data_path_exp)
    data_exp.normalize([list(range(len(data_exp)))])
    data_sim = Data(data_path_sim)
    data_sim.normalize([list(range(len(data_sim)))])
    print('completed', time()-t,'sec')
    loader = get_loader(data_exp,batch_size=batch_size,idx_sets=[list(range(len(data_exp)))])[0]
    #build model
    model = NN()
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    os.makedirs('predict',exist_ok=True)
    outputs = []
    model.load_state_dict(torch.load('weights/W.pth.tar'))
    output = use_model(loader,model,0)
    output = data_sim.revert_normalize(output)
    # output = data.revert_normalize(output)
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    plt.plot(output)
    # json.dump([mpids,outputs,target,std],open('predict/Perov_All.json','w'))
# %%

import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
fig, ax = plt.subplots(1,5)
output = np.reshape(output, (5, 38, 10))
x = np.arange(0, 38)
y = np.arange(0, 10)
x_new = np.linspace(0, 37, 190)
y_new = np.linspace(0, 9, 50)

for i in range(5):
    # Interpolate the data at the new x and y values
    interp = RegularGridInterpolator((x, y), output[i], method='cubic')
    xv, yv = np.meshgrid(y_new, x_new, indexing='ij')
    points = np.column_stack((yv.ravel(), xv.ravel()))
    z_new = interp(points)
    z_new = z_new.reshape(xv.shape)
    ax[i].imshow(z_new.transpose(), cmap='RdBu', interpolation='nearest', vmin=-0.1, vmax=.1)
    ax[i].set_title('layer %d'%(i+1))
    ax[i].axis('off')
plt.show()
# %%
