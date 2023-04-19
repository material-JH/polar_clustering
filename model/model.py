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
    
    
    
    
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         # 50*50
#         self.cnn1 = nn.Conv2d(1,32,kernel_size=5,stride=2,padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.cnn2 = nn.Conv2d(32,64,kernel_size=5,stride=2,padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.cnn3 = nn.Conv2d(64,128,kernel_size=5,stride=2,padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.cnn4 = nn.Conv2d(128,256,kernel_size=2)
#         self.act = nn.Softplus()
        
#         self.fc = nn.Linear(256, 1)
        
#     def forward(self, x):
#         x = self.cnn1(x)
#         x = self.bn1(x)
#         x = self.act(x)
#         x = self.cnn2(x)
#         x = self.bn2(x)
#         x = self.act(x)
#         x = self.cnn3(x)
#         x = self.bn3(x)
#         x = self.act(x)
#         x = self.cnn4(x)
#         x = x[:,:,0,0]
#         x = self.act(x)
#         y = self.fc(x)
#         y = y[:,0]
#         return y