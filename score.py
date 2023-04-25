# import stuff
import argparse
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

from fno import *
from sde import * 
from priors import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
trainset = torchvision.datasets.MNIST(root='', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

def get_grid(sde, input_channels, input_height,prior, n=4, num_steps=100, transform=None, 
             mean=0, std=1, clip=True):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = prior.sample([num_samples, 1, input_height, input_height])
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, lmbd = 0.)
            sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
            epsilon = prior.sample(y0.shape)
            y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon
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
            im.save("samples_"+str(res)+"seed"+str(seed)+"/{}.jpeg".format(i*100+j)) #archetype+

        
def training(seed, prior):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)

    drift_q = model

    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in drift_q.parameters()))
    pool = torch.nn.AvgPool2d(2)


    inf_sde = VariancePreservingSDE(prior, alpha_min=0.1, alpha_max=20.0, T=T )
    gen_sde = PluginReverseSDE(prior, inf_sde, drift_q, T, vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(gen_sde.parameters(), lr=1e-4)

    gen_sde.train()
    
    for ep in range(args.n_epochs):
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
    grid = get_grid(gen_sde, 1, args.input_height, prior, n=4,
                          num_steps=200, transform=None)
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('sample16_grid'+str(seed)) 
    plt.figure()
    grid = get_grid(gen_sde, 1, 32, prior, n=4,
                          num_steps=200, transform=None) #output height?
    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()
    for img, ax in zip(grid,axs):
        ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
    plt.savefig('sample32_grid'+str(seed)) 
            

    save_from_model(gen_sde, 16, seed, prior)
    save_from_model(gen_sde, 32, seed, prior)
    

    return gen_sde

def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior()
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        return ImplicitConv()
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")

if __name__ == '__main__':

    model = FNO2d(8,8,64).to(device) #add u-net too

    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--n_epochs', type=int, required=True, help='ADAM epoch')
    parser.add_argument('--input_height', type=int, required=True, help='starting image dimensions')
    parser.add_argument('--prior', type=choose_prior, required=True, help="Choose between prior setup")

    args = parser.parse_args()
    
    # starting dimensions
    input_channels = 1
    #args.input_height = 16
    #n_epochs = 2
    dimx = input_channels * args.input_height ** 2

    training(3, args.prior)
    training(1, args.prior)
    training(2, args.prior)