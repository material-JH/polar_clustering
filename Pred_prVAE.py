#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi
from model.model import *
from lib.plot import *
#%%
data_post_exp = np.load('output/set1_Ru_002.npy')
data_post_exp = np.stack((data_post_exp, np.load('output/set1_Ru_m002.npy')), axis=-3)
data_post_exp4 = np.load('output/set4_SRO_002.npy')
data_post_exp4 = np.stack((data_post_exp4, np.load('output/set4_SRO_m002.npy')), axis=-3)
# data_post_exp2 = np.load('output/set2_Ru_002.npy')
# data_post_exp3 = np.load('output/set3_SRO_002.npy')

#%%

mean = np.mean(data_post_exp)
std = np.std(data_post_exp)
data_post_exp = (data_post_exp - mean) / std

data_stack = data_post_exp.reshape(-1, 2, data_post_exp.shape[-2], data_post_exp.shape[-1])

mean4 = np.mean(data_post_exp4)
std4 = np.std(data_post_exp4)
data_post_exp4 = (data_post_exp4 - mean4) / std4

data_stack4 = data_post_exp4.reshape(-1, 2, data_post_exp4.shape[-2], data_post_exp4.shape[-1])
#%%
simulations_sep = np.load('output/disk_002_dft.npz')
#%%
polarization_keys_sim = [float(k.split('_')[2]) for k in simulations_sep.keys()]
polarization_keys_sim = np.array(polarization_keys_sim)
# p_mean = np.mean(polarization_keys_sim)
# p_std = np.std(polarization_keys_sim)
# polarization_keys_sim = (polarization_keys_sim - p_mean) / p_std
#%%
# Intitialize rVAE model

input_dim = data_stack.shape[2:]
#%%
filename = 'weights/prvae2_002_norm_47_2'
import gc
gc.collect()
rvae = prVAE2(input_dim, latent_dim=10,
                        numlayers_encoder=2, numhidden_encoder=256,
                        numlayers_decoder=2, numhidden_decoder=256,
                        include_reg=True, include_div=True, include_cont=True,
                        reg_weight=.1, div_weight=.1, cont_weight=10, filename=filename)
#%%
if os.path.exists(f'{filename}.tar'):
    rvae.load_weights(f'{filename}.tar')
    print('loaded weights')
#%%

p = rvae.compute_p(torch.tensor(data_stack).float().to(rvae.device))
p4 = rvae.compute_p(torch.tensor(data_stack4).float().to(rvae.device))

print(p.min(), p.max())
print(p4.min(), p4.max())
#%%
ps = p.cpu().detach().numpy()
ps = np.reshape(ps, (5, 38, 10))
fig, ax = plt.subplots(1, 5, figsize=(10, 10))
vmin = -0.13
vmax = 0.03
for i in range(5):
    pcm = ax[i].imshow(ps[i], vmin=vmin, vmax=vmax, cmap='RdBu')
    ax[i].set_title(f'{i - 2}V')
    ax[i].axis('off')
    ax[i].set_aspect('equal')
    # ax[i].set_fontfamily('serif')

# fig.colorbar(pcm, ax=ax[i], orientation='vertical', fraction=.2)

# %%

def p_loss(recon_loss: str,
              in_dim: Tuple[int],
              x: torch.Tensor,
              x_reconstr: torch.Tensor,
              y: torch.Tensor,
              p: torch.Tensor,
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

    p_weight = kwargs.get("p_weight", 1)

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
    is_inf = torch.isinf(y)

    
    p_mean = torch.mean((p[~is_inf] - y[~is_inf]) ** 2)
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
    return likelihood - kl_div - p_mean * p_weight

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
