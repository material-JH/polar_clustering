#%%
import os
from time import time
import numpy as np

import torch
import torch.nn as nn

from model.model import CNN1
from model.data import DataZ, get_loader

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
    data_path_sim='output/z33.npz'
    data_path_exp='output/z13.npy'

    # Best Hyperparameters
    #var. for dataset loader
    batch_size = 512
    #var for training
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    # data = DataExp(data_path)
    data_exp = DataZ(data_path_exp)
    data_exp.normalize([list(range(len(data_exp)))])
    data_sim = DataZ(data_path_sim, 2)
    data_sim.normalize([list(range(len(data_sim)))])
    print('completed', time()-t,'sec')
    loader = get_loader(data_exp,batch_size=batch_size,idx_sets=[list(range(len(data_exp)))])[0]
    #build model
    model = CNN1()
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    os.makedirs('predict',exist_ok=True)
    outputs = []
    model.load_state_dict(torch.load('weights/W.pth.tar'))
    output = use_model(loader,model,0)
    # output = data_sim.revert_normalize(output)
    # output = data.revert_normalize(output)
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    plt.plot(output)
    # json.dump([mpids,outputs,target,std],open('predict/Perov_All.json','w'))
# %%
fig, ax = plt.subplots(1,5)
output_reshape = np.reshape(output, (5, 38, 10))
for n, img in enumerate(output_reshape):
    ax[n].imshow(img, cmap='RdBu', interpolation='bessel')
    ax[n].axis('off')
# %%

fig, ax = plt.subplots(5,1, figsize=(5, 10))
for n, img in enumerate(output_reshape):
    ax[n].plot(np.mean(img, axis=1))
    ax[n].tick_params(labelbottom=False)
    ax[n].set_ylim(-1, 1)
    # ax[n].axis('off')
# %%

z13 = data_exp.raw_Xs[:,0,4].reshape(5, 38, 10)

fig, ax = plt.subplots(1,5)
for n, img in enumerate(z13):
    ax[n].imshow(img, cmap='RdBu', interpolation='nearest')
    ax[n].axis('off')
# %%
