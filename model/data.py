#%%
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

def get_loader(dataset, collate_fn=default_collate,
                batch_size=64, idx_sets=None,
                num_workers=0, pin_memory=False):
    loaders = []
    for idx in idx_sets:
        loaders.append(DataLoader(dataset, batch_size=batch_size,
                       sampler=SubsetRandomSampler(idx),
                       num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=pin_memory))
    return loaders

class DataExp(Dataset):
    def __init__(self, data_path):
        # read csv file
        data = np.load(data_path)
        
        Xs = data.reshape(-1,1,50,50)
        
        Xs = np.concatenate(Xs,axis=0)[:,None,:,:]
        
        # save it to the object
        self.raw_Xs = Xs
        
    def normalize(self,train_idxs):
        self.X_mu = np.mean(self.raw_Xs[train_idxs,:,:,:])
        self.X_std = np.std(self.raw_Xs[train_idxs,:,:,:])
        
        self.Xs = (self.raw_Xs - self.X_mu)/self.X_std
        
        self.Xs = torch.Tensor(self.Xs)
        
    def __len__(self):
        return len(self.raw_Xs)

    def __getitem__(self, idx):
        x = self.Xs[idx,:,:]
        return idx, x
        
class Data(Dataset):
    def __init__(self, data_path):
        # read csv file
        data = np.load(data_path)
        
        Xs = []
        Ys = []
        
        for k, v in data.items():
            Ys.append(float(k.split('_')[2]))
            Xs.append(v)
        
        Xs = np.array(Xs)[:,None,:,:]
        Ys = np.array(Ys)
        
        # save it to the object
        self.raw_Ys = Ys
        self.raw_Xs = Xs
        
    def normalize(self,train_idxs):
        self.X_mu = np.mean(self.raw_Xs[train_idxs,:,:,:])
        self.X_std = np.std(self.raw_Xs[train_idxs,:,:,:])
        self.Y_mu = np.mean(self.raw_Ys[train_idxs])
        self.Y_std = np.std(self.raw_Ys[train_idxs])
        
        self.Xs = (self.raw_Xs - self.X_mu)/self.X_std
        self.Ys = (self.raw_Ys - self.Y_mu)/self.Y_std
        
        self.Xs = torch.Tensor(self.Xs)
        self.Ys = torch.Tensor(self.Ys)
        
    def revert_normalize(self, Y):
        Y = Y*self.Y_std + self.Y_mu
        return Y

    def __len__(self):
        return len(self.raw_Ys)

    def __getitem__(self, idx):
        y = self.Ys[idx]
        x = self.Xs[idx,:,:]
        return idx, y,x

if __name__ == "__main__":
    data = Data('../output/disk_002_dft.npz')
    
# import torchvision

# class Data(Dataset):
#     def __init__(self):
#         # read csv file
#         data = torchvision.datasets.MNIST('./mnist',download=True)
        
#         Xs = []
#         Ys = []
#         for x,y in data:
#             Xs.append(np.asarray(x)[None,None,:,:])
#             Ys.append(y)
            
#         Xs = np.concatenate(Xs,axis=0)
#         Ys = np.array(Ys)
#         # save it to the object
#         self.raw_Ys = Ys
#         self.raw_Xs = Xs
        
#     def normalize(self,train_idxs):
#         self.X_mu = np.mean(self.raw_Xs[train_idxs,:,:,:])
#         self.X_std = np.std(self.raw_Xs[train_idxs,:,:,:])
#         self.Y_mu = np.mean(self.raw_Ys[train_idxs])
#         self.Y_std = np.std(self.raw_Ys[train_idxs])
        
#         self.Xs = (self.raw_Xs - self.X_mu)/self.X_std
#         self.Ys = (self.raw_Ys - self.Y_mu)/self.Y_std
        
#         self.Xs = torch.Tensor(self.Xs)
#         self.Ys = torch.Tensor(self.Ys)
        
#     def __len__(self):
#         return len(self.raw_Ys)

#     def __getitem__(self, idx):
#         y = self.Ys[idx]
#         x = self.Xs[idx,:,:]
#         return idx, y,x
        