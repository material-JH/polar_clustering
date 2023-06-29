#%%
import os
import json
from time import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from model.model import CNN1, FNN 
from model.data import DataHousing, DataZFlatten, Datadia, get_loader, DataZ

def use_model(data_loader, model, criterion, optimizer, epoch, mode, name = None):
    assert mode in ['train','predict']
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    #switch to train model
    if mode == 'train':
        model.train()
    elif mode == 'predict':
        model.eval()

    t0 = time()
    outputs = []
    targets = []
    idxss = []
    for i, (idxs, ys,xs) in enumerate(data_loader):
        targets += ys.cpu().tolist()
        idxss += idxs
        # move input to cuda

        if next(model.parameters()).is_cuda:
            xs = xs.to(device='cuda')
            ys = ys.to(device='cuda')
        #compute output
        if mode == 'train':
            output = model(xs)
            outputs += output.detach().cpu().tolist()
        elif mode == 'predict':
            with torch.no_grad():
                output = model(xs)
            outputs += output.cpu().tolist()
        
        loss = criterion(output, ys)
        #measure accuracy

        losses.update(loss.data.cpu().item(), ys.size(0))
        
        if mode == 'train':
            #backward operation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #measure elapsed time
        batch_time.update(time() - t0)
        t0 = time()
        
        if mode == 'train':
            s = 'Epoch'
        else:
            s = 'Pred '
        
        if name is not None:
            s += ' '+ name
        
        # print(s+': [{0}][{1}/{2}]\t'
        #  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #  epoch, i, len(data_loader), batch_time=batch_time,
        #  loss=losses))
    print(s+' end: [{0}]\t'
      'Time {batch_time.sum:.3f}\t'
      'Loss {loss.avg:.4f}'.format(
      epoch, batch_time=batch_time,
      loss=losses))
    return outputs,targets,idxss

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

# if __name__ == '__main__':
################################ Input ####################################
# data
data_path='output_pnu/z3.npz'
TrainValTeSplitst = [0.8, 0.1, 0.1]
# Model 

# Training
batch_size = 2 ** 8
lr = 0.0004
epochs = 1000
cuda = True
seed = 1234
###########################################################################

# Loading data
print('loading data...',end=''); t = time()
#data = Datadia()
data = DataZFlatten(data_path, 2)



print('completed', time()-t,'sec')

# Make a split
## number of train and validation
ntrain = int(len(data)*TrainValTeSplitst[0])
nval = int(len(data)*TrainValTeSplitst[1])
## randomize
idxs = list(range(len(data)))
random.seed(seed)
random.shuffle(idxs)

## split index
train_idx = idxs[:ntrain]
val_idx = idxs[ntrain:ntrain+nval]
test_idx = idxs[ntrain+nval:]

## normalize
data.normalize(train_idx)
json.dump(data.GetNormParameters(),open('norm_params.json','w'))
# plt.plot(data.Xs[:,1],data.Ys,'.')


## get data loader
train_loader, val_loader, test_loader = get_loader(data,
    batch_size=batch_size,idx_sets=[train_idx,val_idx,test_idx],pin_memory=True)
json.dump([train_idx,val_idx,test_idx],open('split.json','w'))
#build model

# model = CNN1()
model = FNN(data.Xs.shape[1], h_dim=256)
if cuda:
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0])
    model.cuda()

## Training
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr,weight_decay=1e-3)
os.makedirs('weights',exist_ok=True)

# if os.path.exists('weights/W.pth.tar'):
#     model.load_state_dict(torch.load('weights/W.pth.tar'))
#     print('Weights loaded.')

bestval_loss = float('inf')
t0 = time()
losses = []
for epoch in range(epochs):
    output,target,idxs = use_model(train_loader,model,criterion,optimizer,epoch,'train')
    print('Train loss score [%d]:'%epoch, criterion(torch.Tensor(output),torch.Tensor(target)))
    output,target,idxs = use_model(val_loader,model,criterion,optimizer,epoch,'predict','Val')
    val_loss = criterion(torch.Tensor(output),torch.Tensor(target))
    losses.append(val_loss)
    print('Val loss score [%d]:'%epoch, val_loss, end=' ')
    if val_loss < bestval_loss:
        bestval_loss = val_loss
        print('<-Best')
        torch.save(model.state_dict(),'weights/W.pth.tar')
    else: print('')
    #scheduler.step()

print('--------Training time in sec-------------')
print(time()-t0)
print('Testing. Loading best model')
model.load_state_dict(torch.load('weights/W.pth.tar'))
output,target,idxs = use_model(test_loader,model,criterion,optimizer,epoch,'predict','Test')
print('Predict loss score:', criterion(torch.Tensor(output),torch.Tensor(target)))

# save test result
os.makedirs('tests',exist_ok=True); 
idx = np.argsort(idxs)
testoutput = np.array(output)[idx].tolist()
json.dump(testoutput,open('tests/Ypred.json','w'))

plt.plot(losses)
# %%
import matplotlib.pyplot as plt

plt.imshow(data.raw_Xs[300][0])
# %%
import matplotlib.pyplot as plt
with torch.no_grad():
    output,target,idxs = use_model(val_loader,model,criterion,optimizer,epoch,'predict','Val')
    # output = data.revert_normalize(np.array(output))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist2d(target, output,bins=100, cmap='Blues')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')