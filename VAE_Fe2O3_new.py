#%%
import numpy as np
import matplotlib.pyplot as plt
from lib.main import *
import atomai as aoi

#%%
data_exp = np.load('output/Fe2O3.npy')
data_exp = data_exp.astype(np.float32)
initial_shape = data_exp.shape
abf = np.mean(data_exp, axis=(-1, -2))
abf = abf - abf.min()
abf = abf / abf.max()
#%%
class VAEmodel:

    def __init__(self, input_dim, data, ind) -> None:
        self.vae = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[2, 2],
                        numlayers_encoder=2, numhidden_encoder=128,
                        numlayers_decoder=2, numhidden_decoder=128,
                        loss='mse')
        self.data = data.reshape((-1, *data.data.shape[-2:]))
        self.ind = ind
        self.trainSet = self.data[np.sort(np.random.choice(self.data.shape[0], self.data.shape[0] // 2, replace=False))]
        self.testSet = self.data[np.sort(np.setdiff1d(ind, self.trainSet))]

    def normalize(self):
        self.trainSet = fn_on_resized(self.trainSet, normalize_Data)
        self.testSet = fn_on_resized(self.testSet, normalize_Data)
        # self.trainSet -= self.trainSet.mean(axis=(-2, -1), keepdims=True)
        # self.trainSet /= self.trainSet.std(axis=(-2, -1), keepdims=True)
        # self.testSet -= self.testSet.mean(axis=(-2, -1), keepdims=True)
        # self.testSet /= self.testSet.std(axis=(-2, -1), keepdims=True)

    def fit(self, cycle=100):
        self.vae.fit(
            X_train=self.trainSet,
            X_test=self.testSet,
            training_cycles=cycle,
            batch_size=2 ** 8)
    
    def getSeletedData(self, n):
        _, _, alpha = self.vae.encode(self.data[self.ind])
        select_data = alpha[:,n] > 0.5
        return self.ind[select_data]
    
def viewSelectedData(vaemodel, n):
    selected_ind = vaemodel.getSeletedData(n)
    img = np.zeros(initial_shape[0] * initial_shape[1])
    img[selected_ind] = 1
    img = img.reshape(initial_shape[:2])
    plt.imshow(img, alpha=img)
    plt.imshow(abf, alpha=1-abf, cmap='gray')
    plt.show()
# %%

for n in range(10):
    if n == 0:
        selected_ind = np.arange(data_exp.shape[0] * data_exp.shape[1])
    model = VAEmodel(data_exp.data.shape[2:], data_exp, selected_ind)
    model.normalize()
    model.fit(100)
    model.vae.save_weights('output/VAE_Fe2O3_{}.h5'.format(n))
    for m in range(4):
        viewSelectedData(model, m)

    selected_ind = model.getSeletedData(int(input('select: ')))
    print(len(selected_ind))

#%%
self.fit(50)
#%%
selected_ind = self.getSeletedData(1)
#%%
vae2 = VAEmodel(data_exp.data.shape[2:], data_exp, selected_ind)
vae2.normalize()
# %%
vae2.fit(50)
# %%
viewSelectedData(vae2, 1)
#%%
selected_ind2 = vae2.getSeletedData(1)
vae3 = VAEmodel(data_exp.data.shape[2:], data_exp, selected_ind2)
vae3.normalize()
# %%
vae3.fit(50)
# %%
selected_ind3 = vae3.getSeletedData(1)
vae4 = VAEmodel(data_exp.data.shape[2:], data_exp, selected_ind3)
vae4.normalize()
# %%
vae4.fit(50)
# %%
viewSelectedData(vae4, 2)
# %%
