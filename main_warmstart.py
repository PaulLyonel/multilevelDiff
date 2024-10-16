# import stuff
import os
import argparse
import itertools
import numpy as np
import torch
import random
import torchvision
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fno import *
from metrics import *
from sde import *
import time
from priors import *
from utils import get_samples, get_samples_batched, downsampling_fourier
import os
import datetime
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(seed, model, args):
    """
    training the score function

    :param seed: random seed, ensure reproducibility within the same seed
    :param model: function approximator for the score function
    :param args: hyperparameters
    :return:
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Define a transform to normalize the data
  
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(64), transforms.GaussianBlur(7, 0.1)])
    # Download and load the training data
    trainset = datasets.MNIST(root='', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = datasets.MNIST(root='', train=False, download=True, transform=transform)
    print(len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in model.parameters()))

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-4)
    pool = torch.nn.AvgPool2d(2)

    rev_sde.train()
    total_time = 0.

    loss_curve = []
    time_curve = []

    min_loss = 10000.
    for ep in range(args.n_epochs):
        mean_loss = 0.
        for k,(x,y) in enumerate(testloader):
            x = x.to(device)

            with torch.no_grad():
                rev_sde.eval()
                true_loss =  rev_sde.dsm(x).mean().item()*x.shape[0]
                mean_loss = mean_loss + true_loss
        mean_loss = mean_loss/len(testset)
        print('EPOCH: ', ep)
        print(mean_loss)
        if ep == args.n_epochs//2:
            rev_sde.eval()

            with torch.no_grad():

               
                samples = get_samples(rev_sde, 1, 64, 1000, 16)[0]
                plt.figure()
                fig, axs = plt.subplots(4, 4, figsize=(12, 12))
                axs = axs.flatten()
                for img, ax in zip(samples, axs):
                    ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
                plt.tight_layout()

                plt.savefig(str(args.out_dir)+'/halfsamples64'+str(args.prior_name)+str(args.warm_start))
                plt.close()
                samples = get_samples(rev_sde, 1, 128, 1000, 16)[0]
                plt.figure()
                fig, axs = plt.subplots(4, 4, figsize=(12, 12))
                axs = axs.flatten()
                for img, ax in zip(samples, axs):
                    ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
                plt.tight_layout()

                plt.savefig(str(args.out_dir)+'/halfsamples128'+str(args.prior_name)+str(args.warm_start))
                plt.close()
            
        loss_curve.append(mean_loss)          
        time_curve.append(total_time)
        t0 = time.time()
        for k,(x,y) in enumerate(trainloader):
            rev_sde.train()
            x = x.to(device)
            optim.zero_grad()
            if args.warm_start:
                if ep < args.n_epochs//2:
                    loss1 = rev_sde.dsm(downsampling_fourier(x)).mean()
                    loss1.backward()
                else:
                    loss2 = rev_sde.dsm(x).mean()
                    loss2.backward()
            else:
                    loss2 = rev_sde.dsm(x).mean()
                    loss2.backward()
            optim.step()
        t1 = time.time()
        print(t1-t0)
        total_time += (t1-t0)


    print("total time")
    print(total_time)
    rev_sde.eval()

    with torch.no_grad():

        samples = get_samples(rev_sde, 1, 64, 1000, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
        plt.tight_layout()

        plt.savefig(str(args.out_dir)+'/samples64'+str(args.prior_name)+str(args.warm_start))
        plt.close()
        samples = get_samples(rev_sde, 1, 128, 1000, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
        plt.tight_layout()

        plt.savefig(str(args.out_dir)+'/samples128'+str(args.prior_name)+str(args.warm_start))
        plt.close()
    return loss_curve, time_curve


def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior(k1=args.modes_prior,k2=args.modes_prior//2)
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        return ImplicitConv(scale=20)
    elif string.lower() == "bessel":
        return BesselConv(scale = 100, power = 0.55)
    elif string.lower() == "combined_conv":
        return CombinedConv(k1=args.modes_prior,k2=args.modes_prior,scale=10., scale2 = 0.5)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--n_epochs', type=int, default=101, help='epochs of pde training')
    parser.add_argument('--lr', type=float,default=6e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--input_height', type=int, default=32, help='training image resolution')
    parser.add_argument('--prior_name', type=str, default='combined_conv', help="prior setup")
    parser.add_argument('--width', type=str, default=32, help="prior setup")
    parser.add_argument('--modes', type=int, default=14, help='cutoff modes in FNO')
    parser.add_argument('--modes_prior', type=int, default=10, help='cutoff modes in FNO')
    parser.add_argument('--warm_start', type=bool, default=True, help='train or eval')

    parser.add_argument('--seed', type=int, default=0, help='seed for random number generator')
    parser.add_argument('--out_dir', type=str, default='warmstart_results', help='directory for result')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.prior = choose_prior(args.prior_name)
   
    model = FNO2d(args.modes,args.modes,args.width)

    input_channels = 1
    loss_curve, time_curve = training(args.seed, model, args)
    if args.warm_start:
        np.save('loss_curve_warm',np.array(loss_curve))
        np.save('time_curve_warm',np.array(time_curve))
    else:
        np.save('loss_curve_cold',np.array(loss_curve))
        np.save('time_curve_cold',np.array(time_curve)) 

    plt.figure()
    plt.plot(loss_curve, color = 'red')

    plt.xlabel('Training Epochs')
    plt.ylabel('DSM Loss')
    plt.tight_layout()

    plt.savefig(str(args.out_dir)+'/loss_curve'+str(args.prior_name))
    plt.close()
