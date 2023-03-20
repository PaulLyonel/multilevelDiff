# import stuff
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from fno import *
import matplotlib.pyplot as plt
from sde_fno import * 


device = 'cuda'

# starting dimensions
input_channels = 1
input_height = 16
n_epochs = 300

dimx = input_channels * input_height ** 2


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
trainset = torchvision.datasets.MNIST(root='', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)


def get_grid(sde, input_channels, input_height,prior, n=4, num_steps=100, transform=None, 
             mean=0, std=1, clip=True):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, 1, input_height, input_height).to(sde.T)
    y0 = prior(y0)

    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, prior,lmbd = 0.)

            sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
            epsilon = torch.randn(y0.shape[0],1,y0.shape[2],y0.shape[3], device = device)
            epsilon = prior(epsilon)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * epsilon

    return y0

def save_from_model(gen_sde,res, seed, prior):
    for i in range(100):
        with torch.no_grad():
            sampled_images = get_grid(gen_sde, 1, res, prior,n=10,
                          num_steps=200, transform=None)


        for j in range(sampled_images.shape[0]):
            sample_image = torch.clamp(sampled_images[j],0.,1.)
            arr = sample_image.cpu().data.numpy()*255
            arr = arr.astype(np.uint8).squeeze(0)

            im = Image.fromarray(arr)
            im.save("samples_fno"+str(res)+"seed"+str(seed)+"/{}.jpeg".format(i*100+j))

        
def training(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)

    prior = SpectralConv2d(1,1,16,9, rand = False).to(device)

    drift_q = FNO2d(8,8,64).to(device)
    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in drift_q.parameters()))
    pool = torch.nn.AvgPool2d(2)


    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T, prior = prior)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype='Rademacher', debias=False, prior = prior).to(device)
    optim = torch.optim.Adam(gen_sde.parameters(), lr=1e-4)

    gen_sde.train()
    
    for ep in range(n_epochs):
        print('EPOCH:', ep)
        for k,(x,y) in enumerate(trainloader):
            x = x.to(device)       
            loss = gen_sde.dsm(pool(x), prior).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    gen_sde.eval()
    plt.figure()
    grid = get_grid(gen_sde, 1, 16, prior,n=4,
                          num_steps=200, transform=None)
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('fno16_grid'+str(seed))
    plt.figure()
    grid = get_grid(gen_sde, 1, 32, prior,n=4,
                          num_steps=200, transform=None)
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('fno32_grid'+str(seed))
            

    save_from_model(gen_sde, 16, seed, prior)
    save_from_model(gen_sde, 32, seed, prior)
    

    return gen_sde
training(3)
training(1)
training(2)

