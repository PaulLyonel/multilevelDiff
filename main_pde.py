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
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fno import *
from metrics import *
from sde import * 
from priors import *
from utils import get_samples, get_samples_batched #, pooling_fourier
import os
import datetime
import h5py
from torch.utils.data import DataLoader
from metrics import sw_approx
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pooling_fourier(input_signal):
    k = 2 #kernel size
    input_fft = torch.fft.fft2(input_signal)
    
    kernel = torch.ones((k, k), dtype=input_signal.dtype, device=input_signal.device) / (k * k)
    padded_kernel = torch.zeros_like(input_signal)
    padded_kernel[..., :k, :k] = kernel
    
    kernel_fft = torch.fft.fft2(padded_kernel)
    kernel_fft_normalized = 2 * kernel_fft
    multiplied_fft = input_fft * kernel_fft_normalized
    pooled_signal = torch.fft.ifft2(multiplied_fft).real
    downsampled_output = pooled_signal[..., ::k, ::k]
    
    return downsampled_output

def setup_logger(log_file, output_dir):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
def training(seed, model, output_dir, args):
    """
    training the score function

    :param seed: random seed, ensure reproducibility within the same seed
    :param model: function approximator for the score function
    :param output_dir: directory for result
    :param args: hyperparameters
    :return:
    """
    log_file = "training_log.txt"
    logger = setup_logger(log_file, output_dir)
    logger.info("Starting training with the following parameters:")
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # Download and load the training data
    
    hf = h5py.File('./multilevelDiff/PDEBench/pdebench/data_download/data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5', 'r')
    dataset = torch.empty(0)

    for i in list(hf.keys()):
        data = np.array(hf[str(i)]['data'])[50]
        dataset = torch.cat((dataset, torch.from_numpy(data).float().unsqueeze(0)),0)

    trainsize = 990
    dataset_val = dataset[trainsize:]#1490
    dataset = dataset[:trainsize]


    dataset = dataset[:,:,:,0].squeeze().unsqueeze(1)
    dataset_val = dataset_val[:,:,:,0].squeeze().unsqueeze(1).to(device)
    train_loader = DataLoader(dataset,
             batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val,
             batch_size=args.batch_size, shuffle=True)
    val_samp = next(iter(val_loader))[0]
    logger.info('NUMBER OF PARAMETERS:')
    logger.info(str(sum(p.numel() for p in model.parameters())))
    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in model.parameters()))

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    rev_sde.train()

    loss_curve = []
    loss_curve_high = []
    loss_curve_mid = []
    sw_scores = []
    min_sw = 10

    rev_sde.train()
    for ep in range(args.n_epochs):
        with torch.no_grad():
            rev_sde.eval()
            true_loss =  rev_sde.dsm(dataset_val).mean().item()
            true_loss_res = rev_sde.dsm(pooling_fourier(pooling_fourier(dataset_val))).mean().item()
            true_loss_mid = rev_sde.dsm((pooling_fourier(dataset_val))).mean().item()
            
        loss_curve_high.append(true_loss)
        loss_curve.append(true_loss_res)
        loss_curve_mid.append(true_loss_mid)
        loss_curve.append(true_loss_res)
        for k,x in enumerate(train_loader):
            rev_sde.train()
            x = x.to(device) 
            loss = rev_sde.dsm(pooling_fourier(pooling_fourier(x))).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        if ep % args.vis_freq == 0 or ep == args.n_epochs - 1:
            rev_sde.eval()
            with torch.no_grad():
                logger.info(f"Epoch {ep + 1} Loss: {loss.item()}, True Loss: {true_loss}, True Loss Mid: {true_loss_mid}, True Loss Res: {true_loss_res}")
                gen = torch.empty(0, device = device)
                for k in range(3):
                    samples = get_samples(rev_sde, 1, 128, 200, trainsize//3)[0]
                    gen = torch.cat((gen,samples),0)
                    # print(gen.shape, 'gen', samples.shape, 'samples')

                logger.info('SW')
                sw_value = sw_approx(gen.to('cpu').view(trainsize,128**2), dataset.view(trainsize,128**2))
                logger.info(sw_value)
            
                if sw_value < min_sw:
                    out_file = output_dir + '/model' + str(args.prior_name) + 'modes' + str(args.modes)
                    torch.save(rev_sde, ("%s-min_checkpoint.pt") % (out_file)) #args.out_dir+'/min_checkpoint.pt'
                    min_sw = sw_value

    rev_sde.eval()
    with torch.no_grad():

        samples = get_samples(rev_sde, 1, 32, 200, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy())
        # plt.tight_layout()

        # plt.savefig(str(args.out_dir)+'/samples32'+str(args.prior_name))
            ax.axis('off') 
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        plt.savefig(output_dir+'/samples32'+str(args.prior_name))
        plt.close()


        # samples = get_samples(rev_sde, 1, 64, 200, 16)[0]
        # plt.figure()
        # fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        # axs = axs.flatten()
        # for img, ax in zip(samples, axs):
        #     ax.imshow(img.squeeze(0).cpu().data.numpy())
        #     ax.axis('off') 
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # plt.savefig(output_dir+'/samples64'+str(args.prior_name))
        # plt.close()

        # samples = get_samples(rev_sde, 1, 128, 200, 16)[0]
        # plt.figure()
        # fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        # axs = axs.flatten()
        # for img, ax in zip(samples, axs):
        #     ax.imshow(img.squeeze(0).cpu().data.numpy())
        #     ax.axis('off') 
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # plt.savefig(output_dir+'/samples128'+str(args.prior_name))
        # plt.close()


        # samples = get_samples(rev_sde, 1, 256, 200, 16)[0]
        # plt.figure()
        # fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        # axs = axs.flatten()
        # for img, ax in zip(samples, axs):
        #     ax.imshow(img.squeeze(0).cpu().data.numpy())
        #     ax.axis('off') 
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # plt.savefig(output_dir+'/samples256'+str(args.prior_name))
        # plt.close()

        # samples = dataset[:16]
        # plt.figure()
        # fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        # axs = axs.flatten()
        # for img, ax in zip(samples, axs):
        #     ax.imshow(img.squeeze(0).cpu().data.numpy())
        #     ax.axis('off') 
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # plt.savefig(output_dir+'/truesamples'+str(args.prior_name))
        # plt.close()


        rev_sde = torch.load(out_file+'-min_checkpoint.pt')
        rev_sde.eval()
        samples = get_samples(rev_sde, 1, 32, 200, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy())
            ax.axis('off') 
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        plt.savefig(output_dir+'/bestsamples32'+str(args.prior_name))
        plt.close()


        samples = get_samples(rev_sde, 1, 64, 200, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy())
            ax.axis('off') 
        plt.tight_layout()

        plt.savefig(output_dir+'/bestsamples64'+str(args.prior_name))
        plt.close()

        samples = get_samples(rev_sde, 1, 128, 200, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy())
            ax.axis('off') 
        plt.tight_layout()

        plt.savefig(output_dir+'/bestsamples128'+str(args.prior_name))
        plt.close()


        samples = get_samples(rev_sde, 1, 256, 200, 16)[0]
        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img.squeeze(0).cpu().data.numpy())
            ax.axis('off') 
        plt.tight_layout()

        plt.savefig(output_dir+'/bestsamples256'+str(args.prior_name))
        plt.close()
        with torch.no_grad():
            gen = torch.empty(0, device = device)
            for k in range(3):
                samples = get_samples(rev_sde, 1, 128, 200, 330)[0]
                gen = torch.cat((gen,samples),0)
                # print(gen.shape, 'gen', samples.shape, 'samples')

        logger.info('SW')
        logger.info(sw_approx(gen.to('cpu').view(trainsize,128**2), dataset.view(trainsize,128**2)))
     
    return loss_curve, loss_curve_mid, loss_curve_high


def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior(k1=args.priormodes, k2=args.priormodes//2) 
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        return ImplicitConv(scale=args.scale)
    elif string.lower() == "combined_conv":
        return CombinedConv(k1=args.priormodes, k2=args.priormodes//2, scale=args.scale, power=args.power)
    elif string.lower() == "bessel":
        return BesselConv(scale=args.scale, power=args.power)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--n_epochs', type=int, default=200, help='epochs of pde training')
    parser.add_argument('--lr', type=float,default=1e-3, help='learning rate')
    parser.add_argument('--vis_freq', type=int, default=50, help='visualization frequency')
    parser.add_argument('--model', type=str, default='fno', help='nn model')
    parser.add_argument('--batch_size', type=int, default=8, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--input_height', type=int, default=32, help='training image resolution') #64, 128 - reduce downsampling operation (pooling_fourier) in train function accordingly
    parser.add_argument('--prior_name', type=str, default='bessel', help="prior setup")
    parser.add_argument('--width', type=str, default=32, help="prior setup")
    parser.add_argument('--modes', type=int, default=12, help='cutoff modes in FNO')
    parser.add_argument('--priormodes', type=int, default=32, help='cutoff modes in prior')
    parser.add_argument('--scale', type=float, default=8, help='learning rate')
    parser.add_argument('--power', type=float, default=0.55, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generator')
    parser.add_argument('--out_dir', type=str, default='pde_results', help='directory for result')

    args = parser.parse_args()
    output_dir = f"{args.out_dir}/pm{args.priormodes}_prior{args.prior_name}_lr{args.lr}_scale{args.scale}_power{args.power}_seed{args.seed}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.prior = choose_prior(args.prior_name)
    model = FNO2d(args.modes,args.modes,args.width).to(device) 

    input_channels = 1
    loss_curve, loss_curve_mid, loss_curve_high = training(args.seed, model, output_dir, args)
    plt.figure()
    plt.plot(loss_curve, color = 'red')
    plt.plot(loss_curve_mid, color = 'green')
    plt.plot(loss_curve_high, color = 'blue')
    plt.xlabel('Training Epochs')
    plt.ylabel('DSM Loss')
    plt.tight_layout()

    plt.savefig(str(args.out_dir)+'/loss_curve'+str(args.prior_name))
    plt.close()




