# import stuff
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from fno import *
from sde_lap import * 
import matplotlib.pyplot as plt


device = 'cuda'

# starting dimensions
input_channels = 1
input_height = 16
n_epochs = 10


def compConv(x,fun= lambda x : x):
    """
    compute fun(conv_op(K))*x assuming periodic boundary conditions on x

    where
    K       - is a 2D convolution stencil (assumed to be separable)
    conv_op - means that we build the matrix representation of the operator
    fun     - is a function applied to the operator (as a matrix-function, not
component-wise), default fun(x)=x
    x       - are images, torch.tensor, shape=N x 1 x nx x ny
    """
    n = x.shape
    # Laplacian stencil
    nx = n[2]
    ny = n[3]
    hx = 1.0/nx
    hy = 1.0/ny
    # Laplacian stencil
    K = torch.zeros(3,3)
    K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
    K[0,1] = -1.0/(hy**2)
    K[1,0] = -1.0/(hx**2)
    K[2,1] = -1.0/(hy**2)
    K[1,2] = -1.0/(hx**2)
    K = K.to(device)
    m = K.shape
    mid1 = (m[0]-1)//2
    mid2 = (m[1]-1)//2



    Bp = torch.zeros(n[2],n[3], device = device)
    Bp[0:mid1+1,0:mid2+1] = K[mid1:,mid2:]
    Bp[-mid1:, 0:mid2 + 1] = K[0:mid1, -(mid2 + 1):]
    Bp[0:mid1 + 1, -mid2:] = K[-(mid1 + 1):, 0:mid2]
    Bp[-mid1:, -mid2:] = K[0:mid1, 0:mid2]
    xh = torch.fft.rfft2(x)
    Bh = torch.fft.rfft2(Bp)
    lam = fun(torch.abs(Bh)).to(device)
    xh = xh.to(device)
    lam[torch.isinf(lam)] = 0.0
    xBh = xh * lam.unsqueeze(0).unsqueeze(0)
    xB = torch.fft.irfft2(xBh)
    xB = 9*xB
    return xB,lam

dimx = input_channels * input_height ** 2

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
trainset = torchvision.datasets.MNIST(root='', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

def get_grid(sde, input_channels, input_height, n=4, num_steps=100, transform=None, 
             mean=0, std=1, clip=True):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, 1, input_height, input_height).to(sde.T)
    y0 = compConv(y0, fun = lambda x: (1/torch.sqrt(x)))[0]


    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, lmbd = 0.)

            sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
            epsilon = torch.randn(y0.shape[0],1,y0.shape[2],y0.shape[3], device = device)
            epsilon_add = compConv(epsilon, fun = lambda x: (1/torch.sqrt(x)))[0]

            y0 = y0 + delta * mu + delta ** 0.5 * sigma * epsilon_add

    return y0

def save_from_model(gen_sde,res, seed):
    for i in range(100):
        with torch.no_grad():
            sampled_images = get_grid(gen_sde, 1, res, n=10,
                          num_steps=200, transform=None)


        for j in range(sampled_images.shape[0]):
            sample_image = torch.clamp(sampled_images[j],0.,1.)
            arr = sample_image.cpu().data.numpy()*255
            arr = arr.astype(np.uint8).squeeze(0)

            im = Image.fromarray(arr)
            im.save("samples_lap"+str(res)+"seed"+str(seed)+"/{}.jpeg".format(i*100+j))

        
def training(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)


    drift_q = FNO2d(8,8,64).to(device)
    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in drift_q.parameters()))
    pool = torch.nn.AvgPool2d(2)


    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype='Rademacher', debias=False).to(device)

    optim = torch.optim.Adam(gen_sde.parameters(), lr=1e-4)

    gen_sde.train()
    
    for ep in range(n_epochs):
        print('EPOCH:', ep)
        for k,(x,y) in enumerate(trainloader):
            x = x.to(device)
            x = x - torch.mean(x, dim = 0)
            loss = gen_sde.dsm(pool(x)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    gen_sde.eval()
    plt.figure()
    grid = get_grid(gen_sde, 1, 16,n=4,
                          num_steps=200, transform=None)
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('lap16_grid'+str(seed))
    plt.figure()
    grid = get_grid(gen_sde, 1, 32,n=4,
                          num_steps=200, transform=None)
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('lap32_grid'+str(seed))
            
    save_from_model(gen_sde, 16, seed)
    save_from_model(gen_sde, 32, seed)
    

    return gen_sde

training(1)
training(2)
training(3)
