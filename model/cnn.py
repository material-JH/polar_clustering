import torch.nn as nn
import torch

class encoder(nn.Module):
    def __init__(self,fea_latent = 256,fea = 64):
        super(encoder, self).__init__()
        self.fea = fea
        self.fea_latent = fea_latent

        self.conv1R = nn.Conv2d(1, int(fea/2), kernel_size=4, stride=4, padding=1)
        self.conv2R = nn.Conv2d(int(fea/2), fea, kernel_size=4, stride=4, padding=1)
        self.conv3R = nn.Conv2d(fea, fea, kernel_size=prb_set.Rec_pix_size // (4 * 4), stride=1, padding=0)
        
        self.conv1r = nn.Conv2d(fea, fea, kernel_size=3, stride=1, padding=0)
        self.conv2r = nn.Conv2d(fea, int(fea_latent/2), kernel_size=3, stride=1, padding=0)
        self.conv3r = nn.Conv2d(int(fea_latent/2), fea_latent, kernel_size=3, stride=1, padding=0)
        
        self.act = nn.LeakyReLU(0.2)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # x = R^[B, 7, 7, 32, 32]
        x_shape = torch.Tensor([i for i in x.shape]).to(torch.int)
        x = x.reshape(torch.prod(x_shape[:3]),*x_shape[-2:])[:,None,:,:]
        # reciprocal space convolution
        x = self.conv1R(x)
        x = self.act(x)
        x = self.conv2R(x)
        x = self.act(x)
        x = self.conv3R(x)
        x = self.act(x)
        x = x.reshape(torch.prod(x_shape[:3]),self.fea)
        x = x.reshape(*x_shape[:3],self.fea)
        x = x.swapaxes(3,1)
        # real space convolution
        x = self.conv1r(x)
        x = self.act(x)
        x = self.conv2r(x)
        x = self.act(x)
        x = self.conv3r(x)
        x = self.act2(x)
        x = x.reshape(x_shape[0],self.fea_latent)
        return x
