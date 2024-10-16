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
from utils import get_samples, get_samples_batched, downsampling_fourier, get_samples_true
import os
import datetime
from unet_no_att import UNet

from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def compl_mul2d(input, weights):
    return torch.einsum("bixy,ioxy->boxy", input, weights)

def sample_prior_mixture(prior, batch_size, res):
    samples_prior = prior.sample((batch_size, 1, res,res)).to(device)
    func1 = torch.ones(batch_size,1, res,res, device = device)
    func2 = torch.ones(batch_size,1, res,res, device = device)
    for k in range(res):
        func1[:,:,:,k] *= k/(res)
    for k in range(res):
        func2[:,:,k,:] *= (res-k)/(res)
        
    mean1 = prior.Qmv(func1)
    mean2 = prior.Qmv(func2)
    bern = torch.bernoulli(0.5*torch.ones(batch_size)).to(device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    samples = bern*(mean1)+ (1-bern)*(mean2)+ samples_prior
    return samples

def sample_prior_target(prior, batch_size, res):
    samples_prior = prior.sample((batch_size, 1, res,res)).to(device)
    return samples


def find_matrix(prior,res):
    stack = torch.empty(0, device = device)
    for k in range(res**2):
        noise = torch.zeros(1,1,res**2).to(device)
        noise[:,:,k] = 1.
        conv = args.prior.Qmv(noise.view(1,1,res,res))
        stack = torch.cat((stack, conv.view(1,res**2)),0)

    return stack 

def closed_form_score(sde,matrix, prior, t, x):
    res = x.shape[2]
    matrix = matrix.to(device)
    matrix_inv = torch.linalg.inv(matrix)

    
    
    x = x.requires_grad_()
    var_weight = sde.var_weight(t)
    mean_weight = sde.mean_weight(t)

    func1 = torch.ones(x.shape[0],1, res,res, device = device)
    func2 = torch.ones(x.shape[0],1, res,res, device = device)
    for k in range(res):
        func1[:,:,:,k] *= k/(res)
    for k in range(res):
        func2[:,:,k,:] *= (res-k)/(res)
        
    mean1 = prior.Qmv(func1)
    mean2 = prior.Qmv(func2)

    prod1 = ((x-mean_weight*mean1).view(x.shape[0], res**2))@matrix_inv
    prod2 = ((x-mean_weight*mean2).view(x.shape[0], res**2))@matrix_inv

    vector1 = -0.5*((prod1**2).sum(dim = 1))
    vector2 = -0.5*((prod2**2).sum(dim = 1))
    score =  torch.logsumexp(torch.cat((vector1.unsqueeze(0), vector2.unsqueeze(0)),0), dim = 0)
    grad = torch.autograd.grad(score.sum(), x)[0].view(x.shape[0],res**2)
    Q = matrix@matrix.T
    return (grad@Q).view(x.shape[0],1,res,res)*sde.g(t, x)


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
    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in model.parameters()))

    pool = torch.nn.AvgPool2d(2)

      
    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-4)


    matrix = find_matrix(args.prior, args.input_height)
    
    matrix_large = find_matrix(args.prior, 2*args.input_height)    

    rev_sde.train()
    if args.train ==True:
        loss_curve = []
        loss_curve_high = []
        loss_curve_down = []
        score_list = []
        score_list_high = []

        min_loss = 10000.
        for ep in range(args.n_epochs):
            mean_loss = 0.
            mean_loss_down = 0.
            mean_loss_high = 0.
            mean_score = 0.
            mean_score2 = 0.
            for k in range(50):
                x = sample_prior_mixture(args.prior,args.score_num, 2*args.input_height)
                x_coarse = sample_prior_mixture(args.prior,args.score_num, args.input_height)


                with torch.no_grad():
                    rev_sde.eval()
                    true_loss =  rev_sde.dsm(x_coarse).mean().item()
                    mean_loss = mean_loss + true_loss
                    mean_loss_down =  mean_loss_down + rev_sde.dsm(downsampling_fourier(x)).mean().item()
                    true_loss_high =  rev_sde.dsm(x).mean().item()
                    mean_loss_high = mean_loss_high + true_loss_high

                    #score test
                    t_ = torch.rand([x_coarse.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x_coarse) * fwd_sde.T
                    
                    xt,_,_,_ = rev_sde.base_sde.sample(t_, x_coarse, return_noise=True)
                    score_pred = rev_sde.a(xt, t_.squeeze())
    
                    with torch.enable_grad():
                        score_true = closed_form_score(fwd_sde, matrix, args.prior, t_, xt).detach()

                    score_diff = torch.mean(torch.sum(((score_pred-score_true))**2, dim = (1,2,3)),dim = 0)
                    mean_score += score_diff
                    t_ = torch.rand([x_coarse.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * fwd_sde.T

                    
                    xt,_,_,_ = rev_sde.base_sde.sample(t_, x, return_noise=True)
                    score_pred = rev_sde.a(xt, t_.squeeze())
                    with torch.enable_grad():
                        score_true = closed_form_score(fwd_sde, matrix_large, args.prior, t_, xt).detach()

                    score_diff = torch.mean(torch.sum(((score_pred-score_true))**2, dim = (1,2,3)),dim = 0)
                    mean_score2 += score_diff

            mean_loss = mean_loss/50
            mean_loss_high = mean_loss_high/50
            mean_loss_down = mean_loss_down/50
            mean_score = mean_score/50
            mean_score2 = mean_score2/50


            loss_curve.append(mean_loss)
            loss_curve_high.append(mean_loss_high)
            loss_curve_down.append(mean_loss_down)
            score_list.append(mean_score.item())
            score_list_high.append(mean_score2.item())

            for k in range(10):
                rev_sde.train()
                x = sample_prior_mixture(args.prior,args.batch_size, 2*args.input_height)

                loss = rev_sde.dsm(downsampling_fourier(x)).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()

        rev_sde.eval()
        samples_test = get_samples_true(rev_sde, 1, 32, 1000, 25, matrix, args.prior)
        samples_test_64 = get_samples_true(rev_sde, 1, 64, 1000, 25, matrix_large, args.prior)

        with torch.no_grad():
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples_test, axs):
                ax.set_axis_off()
                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/score_test_samples32')
            plt.close()

            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples_test_64, axs):
                ax.set_axis_off()
                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/score_test_samples64')
            plt.close()


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
            samples = sample_prior_mixture(args.prior,2000, 2*args.input_height)
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()

                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/truesamples64'+str(args.prior_name))
            plt.close()

            samples = sample_prior_mixture(args.prior,2000, args.input_height)
            plt.figure()
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            axs = axs.flatten()

            for img, ax in zip(samples, axs):
                ax.set_axis_off()

                ax.imshow(img.squeeze(0).cpu().data.numpy(), cmap= 'gray')
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/truesamples32'+str(args.prior_name))
            plt.close()
            plt.figure()
            plt.plot(loss_curve, color = 'red')
            plt.plot(loss_curve_high, color = 'blue')
            plt.plot(loss_curve_down, color = 'green')

            plt.xlabel('Training Epochs', fontsize = "x-large")
            plt.ylabel('DSM Loss', fontsize = "x-large")
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/loss_curve'+str(args.prior_name))
            plt.close()

            plt.figure()
            plt.plot(score_list, color = 'red')
            plt.plot(score_list_high, color = 'blue')

            plt.xlabel('Training Epochs', fontsize = "x-large")
            plt.ylabel('Score difference', fontsize = "x-large")
            plt.tight_layout()

            plt.savefig(str(args.out_dir)+'/score_curve'+str(args.prior_name))
            plt.close()
    return loss_curve, loss_curve_high


def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior(k1=args.modes_prior,k2=args.modes_prior//2)
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        return ImplicitConv(scale=10)
    elif string.lower() == "combined_conv":
        return CombinedConv(k1=args.modes_prior,k2=args.modes_prior//2,scale=10., scale2 = 0.5)
    elif string.lower() == "bessel":
        return BesselConv(scale = 100, power = 0.55)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--n_epochs', type=int, default=100, help='epochs of pde training')
    parser.add_argument('--lr', type=float,default=6e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--input_height', type=int, default=32, help='training image resolution')
    parser.add_argument('--prior_name', type=str, default='combined_conv', help="prior setup")
    parser.add_argument('--width', type=str, default=32, help=" setup")
    parser.add_argument('--modes', type=int, default=14, help='cutoff modes in FNO')
    parser.add_argument('--modes_prior', type=int, default=14, help='cutoff modes in FNO')
    parser.add_argument('--train', type=bool, default=True, help='train or eval')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generator')
    parser.add_argument('--score_num', type=int, default=300, help='number of samples for score estimate')

    parser.add_argument('--out_dir', type=str, default='gmm_results', help='directory for result')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.prior = choose_prior(args.prior_name)
    model = FNO2d(args.modes,args.modes,args.width)
    #model = UNet(
    #    input_channels=1,
    #    input_height=args.input_height,
    #    ch=32,
    #    ch_mult=(1, 2),
    #    num_res_blocks=2,
    #    resamp_with_conv=True).to(device)
    input_channels = 1
    loss_curve, loss_curve_high = training(args.seed, model, args)

