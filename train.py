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
# from pytorch_fid.fid_score import calculate_frechet_distance

from fno import *
from unet import *
from sde import * 
from priors import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
trainset = torchvision.datasets.MNIST(root='', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

def get_grid(sde, input_channels, input_height, args, transform=None, 
             mean=0, std=1, clip=True):
    #num_samples = n ** 2
    delta = sde.T / args.num_steps
    y0 = args.prior.sample([args.num_samples, 1, input_height, input_height])
    ts = torch.linspace(0, 1, args.num_steps + 1).to(y0) * sde.T
    ones = torch.ones(args.num_samples, 1, 1, 1).to(y0)

    y1 = args.prior.sample(y0.shape)
    y2 = args.prior.sample(y0.shape)

    with torch.no_grad():
        for i in range(args.num_steps):
            mu = sde.mu(ones * ts[i], y0, lmbd = 0.)
            sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
            epsilon = args.prior.sample(y0.shape)
            y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon
            if i == args.num_steps // 3:
                y2 = y0
            elif i == (args.num_steps * 2) // 3:
                y1 = y0

    return y0, y1, y2

def save_from_model(gen_sde,res, seed, args):
    for i in range(100):
        with torch.no_grad():
            sampled_images = get_grid(gen_sde, 1, res, args.input_height, args, transform=None)
            final_image = sampled_images[0]

        for j in range(final_images.shape[0]):
            sample_image = torch.clamp(final_images[j],0.,1.)
            arr = sample_image.cpu().data.numpy()*255
            arr = arr.astype(np.uint8).squeeze(0)

            im = Image.fromarray(arr)
            im.save("samples_"+str(res)+"seed"+str(seed)+"/{}.jpeg".format(i*100+j)) 


        
def training(seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)

    drift_q = model

    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in drift_q.parameters()))
    pool = torch.nn.AvgPool2d(2)

    inf_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=T )
    gen_sde = PluginReverseSDE(args.prior, inf_sde, drift_q, T, vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(gen_sde.parameters(), lr=args.lr)

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
    grid = get_grid(gen_sde, 1, args.input_height, args, transform=None)
    # final_image = grid[0]
    # lastthird_image = grid[1]
    # firstthird_image = grid[2]

    fig, axs = plt.subplots(4,4, figsize = (12,12))
    axs = axs.flatten()

    for i in range(3):
        for img, ax in zip(grid[i],axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
        plt.savefig(args.model+str(args.prior)[8:11]+'period'+str(i)+str(args.input_height)+'samplegrid'+str(seed)) 
        plt.figure()

    if args.model=='fno':
        grid = get_grid(gen_sde, 1, args.input_height*2, args, transform=None) 
        fig, axs = plt.subplots(4,4, figsize = (12,12))
        axs = axs.flatten()

        for i in range(3):
            for img, ax in zip(grid[i],axs):
                ax.imshow(img.squeeze(0).cpu().data.numpy(), vmin = 0., vmax = 1., cmap = 'gray')
            plt.savefig(args.model+str(args.prior)[8:11]+'period'+str(i)+'sample32_grid'+str(seed)) 
            plt.figure()
            
    if args.save:
        save_from_model(gen_sde, args.input_height, seed, args.prior)
        #save_from_model(gen_sde, args.input_height*2, seed, args.prior)

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

    parser = argparse.ArgumentParser(description='Training arguments')

    parser.add_argument('--n_epochs', type=int, default=300, help='ADAM epoch')
    parser.add_argument('--lr', type=float,default=1e-4, help='ADAM learning rate')
    
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples drawing from the prior')
    parser.add_argument('--num_steps', type=int, default=200, help='number of SDE timesteps')
    parser.add_argument('--input_height', type=int, default=16,  help='starting image dimensions')
    parser.add_argument('--prior', type=choose_prior, required=True, help="prior setup")
    
    parser.add_argument('--model', type=str, required=True,help='nn model') 

    parser.add_argument('--modes', type=int, default=8, help='cutoff modes in FNO')

    parser.add_argument('--save', type=bool, default=False,help='save from model') 
    args = parser.parse_args()

    
    if args.model=='fno':
        model = FNO2d(args.modes,args.modes,64).to(device) #add u-net too
    elif args.model=='unet':
        model = UNet(
            input_channels=1,
            input_height=args.input_height,
            ch=32,
            ch_mult=(1, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16,),
            resamp_with_conv=True,
            dropout=0,
            ).to(device)
            
    # starting dimensions
    input_channels = 1
    dimx = input_channels * args.input_height ** 2

    training(3, args)
    training(1, args)
    #training(2, args)