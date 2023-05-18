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
        # idx, inputs = j
        idx, target, inputs = j
        targets.extend(target.cpu().tolist())
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
    
    #Bs = [Bs[i] for i in idx]
    # return outputs
    return np.array(outputs), np.array(targets)
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
    data_path= './output/z33.npz'
    # data_path= './output/set4_Ru_011.npy'
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
    data = DataZ(data_path, 2)
    data.normalize([list(range(len(data)))])
    print('completed', time()-t,'sec')
    loader = get_loader(data,batch_size=batch_size,idx_sets=[list(range(len(data)))])[0]
    #build model
    model = CNN1()
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    os.makedirs('predict',exist_ok=True)
    outputs = []
    model.load_state_dict(torch.load('weights/W.pth.tar'))
    # output = use_model(loader,model,0)
    output, target = use_model(loader,model,0)
    output = data.revert_normalize(output)
    target = data.revert_normalize(target)
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    plt.hist2d(target, output, bins=100, cmap='Blues')
    # json.dump([mpids,outputs,target,std],open('predict/Perov_All.json','w'))
# %%
output = np.array(output)
output = data.revert_normalize(output)
# %%
output = np.array(output)
plt.scatter(target, output)
# %%
