# import stuff
import os
import argparse
import itertools
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fno import *
from metrics import *
from sde import * 
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
    trainset, val_set = torch.utils.data.random_split(trainset, [55000, 5000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader2 = torch.utils.data.DataLoader(val_set, batch_size=5000, shuffle=True)
    val_test = next(iter(valloader2))[0]
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    testset = datasets.MNIST(root='', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    testloader2 = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
    test_set = next(iter(testloader2))[0]


    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in model.parameters()))

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-4)

    rev_sde.train()
    if args.train ==True:
        loss_curve = []
        loss_curve_high = []
        sw_list = []
        min_loss = 10000.
        for ep in range(args.n_epochs):
            mean_loss = 0.
            mean_loss_high = 0.
            for k,(x,y) in enumerate(valloader):
                x = x.to(device)

                with torch.no_grad():
                    rev_sde.eval()
                    true_loss =  rev_sde.dsm(downsampling_fourier(x)).mean().item()*x.shape[0]
                    mean_loss = mean_loss + true_loss
                    true_loss_high =  rev_sde.dsm(x).mean().item()*x.shape[0]
                    mean_loss_high = mean_loss_high + true_loss_high
            mean_loss = mean_loss/len(val_set)
            mean_loss_high = mean_loss_high/len(val_set)

            if ep > 0:
                if  min_loss > mean_loss:
                    print("new loss")
                    min_loss = mean_loss
                    torch.save(model.state_dict(),'model'+str(args.prior_name))
            print('EPOCH: ', ep)
            print(mean_loss)
            
            loss_curve.append(mean_loss)
            loss_curve_high.append(mean_loss_high)

            if ep%10 ==0:
                with torch.no_grad():
                    rev_sde.eval()
                    test_set = test_set#.cpu().data.numpy()
                    samples = get_samples_batched(rev_sde, 1, 64, 200, 2000).cpu()#.data.numpy()
                    for ll in range(len(samples)):
                        samples[ll] = (samples[ll]-samples[ll].min())/(samples[ll].max()-samples[ll].min())
                    print('SW')
                    sw = sw_approx(samples.view(len(samples),-1), test_set.view(len(test_set),-1))
                    sw_list.append(sw.item())
                    print(sw)

            for k,(x,y) in enumerate(trainloader):
                rev_sde.train()
                x = x.to(device) 
                loss = rev_sde.dsm(downsampling_fourier(x)).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()


        model.load_state_dict(torch.load('model'+str(args.prior_name)))
        rev_sde.eval()
        np.save('sw_list'+str(args.prior_name), np.array(sw_list))
        with torch.no_grad():

            samples = get_samples(rev_sde, 1, 32, 1000, 25)[0]
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()
                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/samples32'+str(args.prior_name))
            plt.close()


            samples = get_samples(rev_sde, 1, 50, 1000, 25)[0]
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()

                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/samples50'+str(args.prior_name))
            plt.close()

            samples = get_samples(rev_sde, 1, 64, 1000, 25)[0]
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()

                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/samples64'+str(args.prior_name))
            plt.close()
            plt.figure()
            plt.plot(loss_curve, color = 'red')
            plt.plot(loss_curve_high, color = 'blue')

            plt.xlabel('Training Epochs', fontsize = "x-large")
            plt.ylabel('DSM Loss', fontsize = "x-large")
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/loss_curve'+str(args.prior_name))
            plt.close()
    else:
        model.load_state_dict(torch.load('model'+str(args.prior_name)))
        rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)

        with torch.no_grad():
            rev_sde.eval()
            samples = get_samples(rev_sde, 1, 64, 1000, 25)[0]

            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()

                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/5x5samples64'+str(args.prior_name))
            test_set = test_set#.cpu().data.numpy()
            samples = get_samples_batched(rev_sde, 1, 64, 200, 10000).cpu()#.data.numpy()
            for ll in range(len(samples)):
                samples[ll] = (samples[ll]-samples[ll].min())/(samples[ll].max()-samples[ll].min())
            print('SW')
            print(sw_approx(samples.view(len(samples),-1), test_set.view(len(test_set),-1)))
            print('diversity score')
            print(compute_vendi_score(samples.cpu().data.numpy().reshape(len(samples),-1)))
            print(compute_vendi_score(test_set.cpu().data.numpy().reshape(len(test_set),-1)))

    return loss_curve, loss_curve_high,sw_list


def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior(k1=args.modes_prior,k2=args.modes_prior//2)
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        return ImplicitConv(scale=20)
    elif string.lower() == "combined_conv":
        return CombinedConv(k1=args.modes_prior,k2=args.modes_prior//2,scale=10., scale2 = 0.5)
    elif string.lower() == "bessel":
        return BesselConv(scale = 8, power = 0.55)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--n_epochs', type=int, default=200, help='epochs of pde training')
    parser.add_argument('--lr', type=float,default=6e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--input_height', type=int, default=32, help='training image resolution')
    parser.add_argument('--prior_name', type=str, default='lap_conv', help="prior setup")
    parser.add_argument('--width', type=str, default=32, help="prior setup")
    parser.add_argument('--modes', type=int, default=14, help='cutoff modes in FNO')
    parser.add_argument('--modes_prior', type=int, default=32, help='cutoff modes in FNO')
    parser.add_argument('--train', type=bool, default=False, help='train or eval')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generator')
    parser.add_argument('--out_dir', type=str, default='mnist_results', help='directory for result')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.prior = choose_prior(args.prior_name)
    model = FNO2d(args.modes,args.modes,args.width)

    input_channels = 1
    loss_curve, loss_curve_high, sw_list = training(args.seed, model, args)

