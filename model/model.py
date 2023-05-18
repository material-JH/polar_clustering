#%%
import torch
import torch.nn as nn

# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         # 50*50
#         self.cnn1 = nn.Conv2d(1,32,kernel_size=5)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=5,stride=2)
#         self.cnn2 = nn.Conv2d(32,64,kernel_size=5)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=5,stride=2)
#         self.cnn3 = nn.Conv2d(64,128,kernel_size=5)
#         self.cnn4 = nn.Conv2d(128,256,kernel_size=3)
#         self.act = nn.Softplus()
        
#         self.fc = nn.Linear(256, 1)
        
#     def forward(self, x):
#         x = self.cnn1(x)
#         x = self.act(x)
#         x = self.maxpool1(x)
#         x = self.cnn2(x)
#         x = self.act(x)
#         x = self.maxpool2(x)
#         x = self.cnn3(x)
#         x = self.act(x)
#         x = self.cnn4(x)
#         x = x[:,:,0,0]
#         x = self.act(x)
#         y = self.fc(x)
#         y = y[:,0]
#         return y
    
    
    
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # 50*50
        self.cnn1 = nn.Conv2d(1,32,kernel_size=5,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cnn2 = nn.Conv2d(32,64,kernel_size=5,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(64,128,kernel_size=5,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cnn4 = nn.Conv2d(128,256,kernel_size=5)
        self.act = nn.Softplus()
        
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.cnn4(x)
        x = x[:,:,0,0]
        x = self.act(x)
        y = self.fc(x)
        y = y[:,0]
        return y
    
    

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        # 50*50
        # self.fc1 = nn.Linear(5, 50)

        self.cnn0 = nn.Conv1d(2,8,kernel_size=5,stride=1,padding=1)
        self.bn0 = nn.BatchNorm1d(8)
        self.cnn1 = nn.Conv1d(8,32,kernel_size=5,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.cnn2 = nn.Conv1d(32,64,kernel_size=5,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(64,128,kernel_size=5,stride=2,padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(128,256,kernel_size=3)
        self.act = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # x = self.fc1(x)
        # x = self.act(x)
        x = self.cnn0(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.cnn4(x)
        x = x[:,:,0]
        x = self.act(x)
        y = self.fc2(x)
        y = y[:,0]
        return y
    
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        # 50*50
        self.fc1 = nn.Linear(96, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 1)
        self.act = nn.ELU()
        
    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    nn = CNN1()
    print(nn.forward(torch.randn(1,1,2)))
# %%
