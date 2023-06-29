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
from atomai.losses_metrics import rvae_loss
from typing import Callable, Optional, Tuple, Type, Union

def beta_loss(z_mean: torch.Tensor, **kwargs: Union[List[float], float]) -> torch.Tensor:
    reg_weight = kwargs.get("reg_weight", 1)
    div_weight = kwargs.get("div_weight", 1)
    cont_weight = kwargs.get("cont_weight", 1)
    include_reg = kwargs.get("include_reg", False)
    include_div = kwargs.get("include_div", False)
    include_cont = kwargs.get("include_cont", False)
    if include_reg:
        reg_loss_1 = torch.norm(z_mean, p=1) / z_mean.shape[0]
        if reg_loss_1 == 0:
            reg_loss_1 = torch.tensor(0.5)
        loss = reg_loss_1 * reg_weight

    if include_div:
        div_loss = 0
        for i in range(len(z_mean)):
            no_zero = torch.where(z_mean[i].squeeze() != 0)[0]
            single = z_mean[i][no_zero]
            div_loss += torch.sum(abs(single.reshape(-1, 1) - single)) / 2.0
        div_loss = div_loss / len(z_mean) * div_weight
        beyond_0 = torch.where(torch.sum(z_mean, axis=1) != 0)[0]
        new_latent = z_mean[beyond_0]
        loss += div_loss

    if include_cont:
        Cont_loss = 0
        for i in beyond_0:
            Cont_loss += sum(F.cosine_similarity(
                z_mean[i].unsqueeze(0), new_latent)) - 1

        Cont_loss = cont_weight * Cont_loss / (2.0 * z_mean.shape[0])
        loss += Cont_loss

    return loss

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
    return p_mean * p_weight

def mmd_loss(z: torch.Tensor, y: torch.Tensor, **kwargs: Union[List[float], float]) -> torch.Tensor:
    """
    Calculates MMD loss
    """
    m_weight = kwargs.get("m_weight", 1)
    is_inf = torch.isinf(y) 
    source = z[~is_inf]
    target = z[is_inf]
    mmd = torch.mean((torch.mean(source, axis=0) - torch.mean(target, axis=0)) ** 2)
    return mmd * m_weight

class regVAE(aoi.models.VAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: Tensor, x_reconstr: Tensor, *args: Tensor, **kwargs) -> Tensor:
        return -beta_loss(args[0], **kwargs) + super().elbo_fn(x, x_reconstr, *args, **kwargs)

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
        return -beta_loss(args[0], **kwargs) + super().elbo_fn(x, x_reconstr, *args, **kwargs)

class prVAE(aoi.models.rVAE):
    
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, translation: bool = True, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = 128
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if len(args) == 2:
            z_mean, z_logsd = args
            p = self.fcl_net(z_mean)
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
            p = self.fcl_net(torch.tensor(z_mean).to(self.device))
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
        self.fcl_net = FCLNet(in_dim, self.h_dim)
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)
        self.fcl_net.to(self.device)

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
    
    def __init__(self, h_dim=256, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, translation: bool = True, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = h_dim
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        return -beta_loss(args[0], **kwargs) + super().elbo_fn(x, x_reconstr, *args, **kwargs)

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
        p = self.fcn_net(z2d_mean)
        ploss = p_loss(y, p, **self.kdict_)
        elbo += ploss
        return elbo
    
    def compute_p(self, x2: torch.Tensor | np.ndarray):
        if isinstance(x2, np.ndarray):
            x2 = torch.from_numpy(x2).float().to(self.device)
        with torch.no_grad():
            z2d_mean = torch.zeros((x2.shape[0], (self.z_dim - 2) * 2)).to(self.device)
            for i in range(2):
                x = x2[:,i,:,:]
                z_mean, z_logsd = self.encoder_net(x)
                z2d_mean[:, i * (self.z_dim - 2): (i + 1) * (self.z_dim - 2)] = z_mean[:, [0, *range(3, self.z_dim)]]
            p = self.fcn_net(z2d_mean)
        return p.detach().cpu().numpy()

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
        self.fcn_net = FCLNet(in_dim, self.h_dim)
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

class FCLNet(nn.Module):
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

class MMDrVAE(aoi.models.rVAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, translation: bool = True, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = 128
        self.p_losses = []
        self.mmd_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, translation, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if len(args) == 2:
            z_mean, z_logsd = args
            p = self.fcn_net(z_mean[:, [0, *range(3, self.z_dim)]])
            pl = p_loss(y, p, **kwargs)
            self.p_losses.append(pl.cpu().detach().numpy())
        ml = mmd_loss(args[0], y, **kwargs)
        self.mmd_losses.append(ml.cpu().detach().numpy())
        return rvae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs) - ml - pl

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

    def compute_p(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            z_mean, z_logsd = self.encode(x)
            p = self.fcn_net(torch.tensor(z_mean[:, [0, *range(3, z_mean.shape[1])]]).to(self.device))
        return p.detach().cpu().numpy()

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
        in_dim = (self.z_dim - 2)
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = FCLNet(in_dim, self.h_dim)
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
        with open(self.filename + '_mmdlosses.txt', 'w') as f:
            for item in self.mmd_losses:
                f.write("%s\n" % item)

class pVAE2(aoi.models.VAE):
    
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        h_dim = kwargs.get("h_dim", 256)
        self.h_dim = h_dim
        self.p_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def forward_compute_elbo(self,
                             x2: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        VAE's forward pass with training/test loss computation
        """
        elbo = 0
        x2 = x2.to(self.device)
        z2d_mean = torch.zeros((x2.shape[0], self.z_dim * 2)).to(self.device)
        for i in range(2):
            x = x2[:,i,:,:]
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
            elbo += self.elbo_fn(x, x_reconstr, z_mean, z_logsd, **self.kdict_)
        if mode == "eval":
            with torch.no_grad():
                p = self.fcn_net(z2d_mean)
                self.p_losses.append(p_loss(y, p, **self.kdict_).cpu().detach().numpy())
                elbo -= p_loss(y, p, **self.kdict_)
        else:
            p = self.fcn_net(z2d_mean)
            # elbo -= p_loss(y, p, **self.kdict_)
            return -p_loss(y, p, **self.kdict_)
        return elbo
   
    def compute_p(self, x2: torch.Tensor | np.ndarray):
        if isinstance(x2, np.ndarray):
            x2 = torch.from_numpy(x2).float().to(self.device)
        with torch.no_grad():
            z2d_mean = torch.zeros((x2.shape[0], (self.z_dim) * 2)).to(self.device)
            for i in range(2):
                x = x2[:,i,:,:]
                z_mean, z_logsd = self.encode(x)
                z2d_mean[:, i * (self.z_dim): (i + 1) * (self.z_dim)] = torch.tensor(z_mean)
            p = self.fcn_net(z2d_mean)
        return p.detach().cpu().numpy()

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
        in_dim = self.z_dim * 2
        # in_dim = self.z_dim - 3 if self.translation else self.z_dim - 2
        self.fcn_net = FCLNet(in_dim, self.h_dim)
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

class pVAE(aoi.models.VAE):

    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = kwargs.get("h_dim", 256)
        self.p_losses = []
        self.m_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)    

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, p, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if len(args) == 2:
            z_mean, z_logsd = args
            pl = -p_loss(y, p, **kwargs)
            ml = -mmd_loss(z_mean, y, **kwargs)
            self.p_losses.append(pl.cpu().detach().numpy())
            self.m_losses.append(ml.cpu().detach().numpy())
        return ml + pl + super().elbo_fn(x, x_reconstr, *args, **kwargs)

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
                p = self.fcl_net(z_mean)
        else:
            x_reconstr = self.decoder_net(z)
            p = self.fcl_net(z_mean)
        return self.elbo_fn(x, x_reconstr, y, p, z_mean, z_logsd, **self.kdict_)

    def pred_p(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        x = x.to(self.device)
        with torch.no_grad():
            z_mean, z_logsd = self.encode(x)
            p = self.fcl_net(torch.tensor(z_mean).to(self.device))
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
        self.fcl_net = FCLNet(in_dim, self.h_dim)
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)
        self.fcl_net.to(self.device)

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

class MMD_regpVAE(aoi.models.VAE):
    def __init__(self, in_dim: int = None, latent_dim: int = 2, nb_classes: int = 0, seed: int = 0, **kwargs: int | bool | str) -> None:
        self.h_dim = kwargs.get("h_dim", 256)
        self.p_losses = []
        self.m_losses = []
        self.b_losses = []
        super().__init__(in_dim, latent_dim, nb_classes, seed, **kwargs)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor, y: torch.Tensor, p, *args: torch.Tensor, **kwargs: Union[list, float, int]) -> torch.Tensor:
        if len(args) == 2:
            z_mean, z_logsd = args
            pl = -p_loss(y, p, **kwargs)
            ml = -mmd_loss(z_mean, y, **kwargs)
            bl = -beta_loss(z_mean, **kwargs)
            self.p_losses.append(pl.cpu().detach().numpy())
            self.m_losses.append(ml.cpu().detach().numpy())
            self.b_losses.append(bl.cpu().detach().numpy())
        return bl + ml + pl + super().elbo_fn(x, x_reconstr, *args, **kwargs)

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
                p = self.fcl_net(z_mean)
        else:
            x_reconstr = self.decoder_net(z)
            p = self.fcl_net(z_mean)
        return self.elbo_fn(x, x_reconstr, y, p, z_mean, z_logsd, **self.kdict_)

    def pred_p(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        x = x.to(self.device)
        with torch.no_grad():
            z_mean, z_logsd = self.encode(x)
            p = self.fcl_net(torch.tensor(z_mean).to(self.device))
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
        self.fcl_net = FCLNet(in_dim, self.h_dim)
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)
        self.fcl_net.to(self.device)

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
