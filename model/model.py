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
from typing import Tuple, List, Union, Optional
from atomai.utils import transform_coordinates
from atomai.losses_metrics import reconstruction_loss, kld_normal, kld_rot
from typing import Callable, Optional, Tuple, Type, Union

class regVAE(aoi.models.VAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: Tensor, x_reconstr: Tensor, *args: Tensor, **kwargs) -> Tensor:
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

class regrVAE(aoi.models.rVAE):
    
    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 translation: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

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
        self.h_dim = 128
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if len(args) == 2:
            z_mean, z_logsd = args
            p = self.fcn_net(z_mean)
            pl = p_loss(y, p, **kwargs)
            self.p_losses.append(pl.cpu().detach().numpy())
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs) + pl

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        
        """
        rVAE's forward pass with training/test loss computation
        """
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
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

        return self.elbo_fn(x, x_reconstr, y, z_mean, z_logsd, **self.kdict_)

    def pred_p(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        x = x.to(self.device)
        with torch.no_grad():
            z_mean, z_logsd = self.encode(x)
            p = self.fcn_net(torch.tensor(z_mean).to(self.device))
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
        in_dim = self.z_dim
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = fcnNet(in_dim, self.h_dim)
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
        self.h_dim = 256
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

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
        self.p_losses.append(ploss.cpu().detach().numpy())
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
        self.fcn_net = fcnNet(in_dim, self.h_dim)
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

    def update_metadict(self):
        super().update_metadict()
        with open(self.filename + '_plosses.txt', 'w') as f:
            for item in self.p_losses:
                f.write("%s\n" % item)

class conv_pVAE(aoi.models.rVAE):
    
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = kwargs.get("numhidden_encoder", 256)
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if y is not None:
            z_mean, z_logsd = args
            p = self.fcn_net(z_mean)
            pl = p_loss(y, p, **kwargs)
            self.p_losses.append(pl.cpu().detach().numpy())
            return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs) + pl
        else:
            return reg_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        VAE's forward pass with training/test loss computation
        """
        x = x.to(self.device)
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
            self.kdict_["num_iter"] += 1
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)
        return self.elbo_fn(x, x_reconstr, y, z_mean, z_logsd, **self.kdict_)

    def compute_p(self, x: torch.Tensor):
        with torch.no_grad():
            z_mean, z_logsd = self.encode(torch.tensor(x).to(self.device))
            p = self.fcn_net(torch.tensor(z_mean).to(self.device))
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
        in_dim = self.z_dim
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = fcnNet(in_dim, self.h_dim)
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

    def update_metadict(self):
        super().update_metadict()
        with open(self.filename + '_plosses.txt', 'w') as f:
            for item in self.p_losses:
                f.write("%s\n" % item)

class fcnNet(nn.Module):
    def __init__(self, in_dim, h_dim=256, slope=0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            torch.nn.LeakyReLU(negative_slope=slope),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            torch.nn.LeakyReLU(negative_slope=slope),
            torch.nn.Linear(h_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:,0]

class EncoderNet(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size=3, stride=1, padding=1, latent_dim: int = 2, *args, **kwargs) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc11 = nn.Linear(hidden_channel, latent_dim)
        self.fc12 = nn.Linear(hidden_channel, latent_dim)
        self._out = nn.Softplus() if kwargs.get("softplus_out") else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.ndim in (2, 3) else x.permute(0, -1, 1, 2)
        x = self.encoder(x).squeeze()
        z_mu = self.fc11(x)
        z_logstd = self._out(self.fc12(x))
        return z_mu, z_logstd

class DecoderNet(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_channel=64, output_channel=1, kernel_size=2, stride=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, output_channel, kernel_size=kernel_size, stride=stride),
            )
    
    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.decoder(z)
        return z

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
        self.cnn1 = nn.Conv2d(1,64,kernel_size=5,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.mx1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(64,64,kernel_size=5,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(64,64,kernel_size=5,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.cnn4 = nn.Conv2d(64,64,kernel_size=5)
        self.act = nn.Softplus()
        
        self.fc = nn.Linear(64, 1)
        
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
    def __init__(self, in_dim, h_dim=96):
        super(FNN, self).__init__()
        # 50*50

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)


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

class testnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    tnet = testnet()
    print(tnet(torch.randn(3, 2, 1, 1)).shape)
    #%%
    enet = EncoderNet(1, 64, latent_dim=5)
    print(enet(torch.randn(3, 1, 64, 64))[0].shape)    

    dnet = DecoderNet()
    print(dnet(torch.randn(10, 2)).shape)
    #%%
    simulations_sep = np.load('../output/disk_002_dft.npz')
    simulations_tot = np.stack(simulations_sep.values(), axis=0)
    data_stack = simulations_tot.reshape(-1, 1, 50, 50)
    # data_stack = torch.tensor(data_stack).float()
    polarization_keys_sim = [float(k.split('_')[2]) for k in simulations_sep.keys()]
    polarization_keys_sim = np.array(polarization_keys_sim)
    p_mean = np.mean(polarization_keys_sim)
    p_std = np.std(polarization_keys_sim)
    polarization_keys_sim = (polarization_keys_sim - p_mean) / p_std
    # %%


    #%%
    #%%
    input_dim = data_stack.shape[-2:]
    prnet = conv_pVAE(input_dim, latent_dim=5, conv_encoder=True, conv_decoder=True, translation=False, seed=0)
    prnet.set_encoder(enet)
    data_train, p_train, data_test, p_test = aoi.utils.data_split(data_stack, polarization_keys_sim, format_out="torch_float")
    # %%
    prnet.fit(X_train=data_train,X_test=data_test, 
              y_train=p_train, y_test=p_test,
              epochs=20, batch_size=64)
    # %%

    with torch.no_grad():
        z_mean, z_logsd = prnet.encode(torch.tensor(data_stack).to(prnet.device))
        p = prnet.fcn_net(torch.tensor(z_mean).to(prnet.device))

    # %%
    import matplotlib.pyplot as plt
    plt.hist2d(polarization_keys_sim, p.cpu().detach().numpy(), bins=50)

    # %%

    data_post_exp = np.load('../output/set1_Ru_002.npy')

    mean = np.mean(data_post_exp)
    std = np.std(data_post_exp)
    data_post_exp = (data_post_exp - mean) / std

    data_stack = data_post_exp.reshape(-1, 1, data_post_exp.shape[-2], data_post_exp.shape[-1])

    with torch.no_grad():
        z_mean, z_logsd = prnet.encode(torch.tensor(data_stack).to(prnet.device))
        exp_p = prnet.fcn_net(torch.tensor(z_mean).to(prnet.device))


    # exp_p = rvae.compute_p(torch.tensor(data_stack4).to(rvae.device))
    fig, ax = plt.subplots(1, 5, figsize=(5,5))
    exp_p = exp_p * p_std + p_mean
    exp_p = exp_p.cpu().detach().numpy()
    exp_p = exp_p.reshape(data_post_exp.shape[:-2])
    for i in range(5):
        ax[i].imshow(exp_p[i], cmap='RdBu')

    # %%
