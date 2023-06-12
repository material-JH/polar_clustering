#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import atomai as aoi
import numpy as np
import torch
from torch import Tensor
from torch import nn
from scipy.stats import norm
import os
from typing import Tuple, List, Union, Optional
from atomai.utils import transform_coordinates
from atomai.losses_metrics import reconstruction_loss, kld_normal, kld_rot, infocapacity
from typing import Callable, Optional, Tuple, Type, Union

class regVAE(aoi.models.VAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: Tensor, x_reconstr: Tensor, *args: Tensor, **kwargs) -> Tensor:
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

class regrVAE(aoi.models.rVAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: Tensor, x_reconstr: Tensor, *args: Tensor, **kwargs) -> Tensor:
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

def reg_loss(recon_loss: str,
              in_dim: Tuple[int],
              x: torch.Tensor,
              x_reconstr: torch.Tensor,
              *args: torch.Tensor,
              **kwargs: Union[List[float], float]
              ) -> torch.Tensor:
    """
    Calculates ELBO
    """
    if len(args) == 2:
        z_mean, z_logsd = args
    else:
        raise ValueError(
            "Pass mean and SD values of encoded distribution as args")
    phi_prior = kwargs.get("phi_prior", 0.1)
    reg_weight = kwargs.get("reg_weight", 1)
    div_weight = kwargs.get("div_weight", 1)
    cont_weight = kwargs.get("cont_weight", 1)
    include_reg = kwargs.get("include_reg", False)
    include_div = kwargs.get("include_div", False)
    include_cont = kwargs.get("include_cont", False)
    phi_logsd = z_logsd[:, 0]
    z_mean, z_logsd = z_mean[:, 1:], z_logsd[:, 1:]
    likelihood = -reconstruction_loss(recon_loss, in_dim, x, x_reconstr).mean()
    kl_rot = kld_rot(phi_prior, phi_logsd).mean()
    kl_z = kld_normal([z_mean, z_logsd]).mean()

    kl_div = (kl_z + kl_rot)
    if include_reg:
        reg_loss_1 = torch.norm(z_mean, p=1) / z_mean.shape[0]
        if reg_loss_1 == 0:
            reg_loss_1 = torch.tensor(0.5)
        kl_div += reg_loss_1 * reg_weight

    if include_div:
        div_loss = 0
        for i in range(len(z_mean)):
            no_zero = torch.where(z_mean[i].squeeze() != 0)[0]
            single = z_mean[i][no_zero]
            div_loss += torch.sum(abs(single.reshape(-1, 1) - single)) / 2.0
        div_loss = div_loss / len(z_mean) * div_weight
        beyond_0 = torch.where(torch.sum(z_mean, axis=1) != 0)[0]
        new_latent = z_mean[beyond_0]
        kl_div += div_loss

    if include_cont:
        Cont_loss = 0
        for i in beyond_0:
            Cont_loss += sum(F.cosine_similarity(
                z_mean[i].unsqueeze(0), new_latent)) - 1

        Cont_loss = cont_weight * Cont_loss / (2.0 * z_mean.shape[0])
        kl_div += Cont_loss

    # if capacity is not None:
    #     kl_div = infocapacity(kl_div, capacity, num_iter=num_iter)
    return likelihood - kl_div


def p_loss(y: torch.Tensor,
            p: torch.Tensor,
            **kwargs: Union[List[float], float]
            ) -> torch.Tensor:
    """
    Calculates p loss
    """
    p_weight = kwargs.get("p_weight", 1)
    is_inf = torch.isinf(y)
    p_mean = torch.mean((p[~is_inf] - y[~is_inf]) ** 2)
    return - p_mean * p_weight

class prVAE(aoi.models.rVAE):
    
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, translation: bool = True, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, p:torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        return p_loss(self.loss, self.in_dim, x, x_reconstr, y, p, *args, **kwargs)

    def forward_compute_elbo(self,
                             x2: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        
        """
        rVAE's forward pass with training/test loss computation
        """
        x = x2[:,0,:,:].unsqueeze(1)
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
            self.kdict_["num_iter"] += 1
        p = self.fcn_net(z_mean[:, 3:])
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        phi = z[:, 0]  # angle
        if self.translation:
            dx = z[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z = z[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z = z[:, 1:]  # image content

        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)

        return self.elbo_fn(x, x_reconstr, y, p, z_mean, z_logsd, **self.kdict_)

    def _check_inputs(self, X_train: np.ndarray, y_train: np.ndarray | None = None, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> None:
        pass

    def set_model(self,
                  encoder_net: Type[torch.nn.Module],
                  decoder_net: Type[torch.nn.Module]
                  ) -> None:
        """
        Sets encoder and decoder models
        """
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        in_dim = self.z_dim - 3
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = fcnNet(in_dim)
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)
        self.fcn_net.to(self.device)

    def _2torch(self,
                X: Union[np.ndarray, torch.Tensor],
                y: Union[np.ndarray, torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Rules for conversion of numpy arrays to torch tensors
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        return X, y

class prVAE2(aoi.models.rVAE):
    
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, translation: bool = True, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)
        self.p_losses = []

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, y, *args, **kwargs)

    def forward_compute_elbo(self,
                             x2: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        
        """
        rVAE's forward pass with training/test loss computation
        """
        elbo = 0
        z2d_mean = torch.zeros((x2.shape[0], (self.z_dim - 2) * 2)).to(self.device)
        for i in range(2):
            x = x2[:,i,:,:]
            x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
            if mode == "eval":
                with torch.no_grad():
                    z_mean, z_logsd = self.encoder_net(x)
            else:
                z_mean, z_logsd = self.encoder_net(x)
                z2d_mean[:, i * (self.z_dim - 2): (i + 1) * (self.z_dim - 2)] = z_mean[:, [0, *range(3, self.z_dim)]]
                self.kdict_["num_iter"] += 1
            z_sd = torch.exp(z_logsd)
            z = self.reparameterize(z_mean, z_sd)
            phi = z[:, 0]  # angle
            if self.translation:
                dx = z[:, 1:3]  # translation
                dx = (dx * self.dx_prior).unsqueeze(1)
                z = z[:, 3:]  # image content
            else:
                dx = 0  # no translation
                z = z[:, 1:]  # image content

            x_coord_ = transform_coordinates(x_coord_, phi, dx)
            if mode == "eval":
                with torch.no_grad():
                    x_reconstr = self.decoder_net(x_coord_, z)
            else:
                x_reconstr = self.decoder_net(x_coord_, z)
            elbo += self.elbo_fn(x, x_reconstr, z_mean, z_logsd, **self.kdict_)
        elbo /= 2
        p = self.fcn_net(z2d_mean)
        ploss = p_loss(y, p, **self.kdict_)
        elbo += ploss
        self.p_losses.append(ploss)
        return elbo

    def compute_p(self, x2: torch.Tensor):
        with torch.no_grad():
            z2d_mean = torch.zeros((x2.shape[0], (self.z_dim - 2) * 2)).to(self.device)
            for i in range(2):
                x = x2[:,i,:,:]
                z_mean, z_logsd = self.encoder_net(x)
                z2d_mean[:, i * (self.z_dim - 2): (i + 1) * (self.z_dim - 2)] = z_mean[:, [0, *range(3, self.z_dim)]]
            p = self.fcn_net(z2d_mean)
        return p

    def _check_inputs(self, X_train: np.ndarray, y_train: np.ndarray | None = None, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> None:
        pass

    def set_model(self,
                  encoder_net: Type[torch.nn.Module],
                  decoder_net: Type[torch.nn.Module]
                  ) -> None:
        """
        Sets encoder and decoder models
        """
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        in_dim = (self.z_dim - 2) * 2
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = fcnNet(in_dim)
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)
        self.fcn_net.to(self.device)

    def _2torch(self,
                X: Union[np.ndarray, torch.Tensor],
                y: Union[np.ndarray, torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Rules for conversion of numpy arrays to torch tensors
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        return X, y

class fcnNet(nn.Module):
    def __init__(self, in_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        slope = 0.5
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.LeakyReLU(negative_slope=slope),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(negative_slope=slope),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(negative_slope=slope),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:,0]

class convVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(1,2,kernel_size=5,stride=1,padding=1)
        self.bn0 = nn.BatchNorm2d(2)
        self.cnn1 = nn.Conv2d(2,2,kernel_size=5,stride=1,padding=1)
        self.cnn2 = nn.Conv2d(2,2,kernel_size=5,stride=1,padding=1)
        self.cnn3 = nn.Conv2d(2,2,kernel_size=5,stride=1,padding=1)
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn0(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.cnn1(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.cnn2(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.cnn3(x)
        # y = x[:,:,0]
        return x

def manifold2d(vae, zmin, zmax, d) -> None:  # use torchvision's grid here
    z_p = np.linspace(zmin, zmax, d)
    z = np.zeros((d, vae.z_dim - 3))
    for i in range(d):
        z[i, -1] = z_p[i]
        print(z[i])
    imgs = vae.decode(z)
    fig, ax = plt.subplots(1, d, figsize=(5, 5 * d))
    for i in range(d):
        ax[i].imshow(imgs[i])
        ax[i].axis("off")


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

        self.fc1 = nn.Linear(18, 96)
        self.fc2 = nn.Linear(96,96)
        self.fc3 = nn.Linear(96, 1)
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(96)


    def forward(self, x):
        # x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x[:,0]

if __name__ == "__main__":
    nn = CNN1()
    print(nn.forward(torch.randn(1,1,2)))
# %%
