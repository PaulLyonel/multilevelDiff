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
from unet_no_att import UNet
from fno import *
from mmd import *
from unet_no_att import *
from sde import * 
from priors import *
from utils import get_samples, makedirs, get_logger, get_samples_batched
import os
import datetime
import utils_patch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(seed, model, args,out_file=None):
    """
    training the score function

    :param seed:
    :param model: function approximator for the score function
    :param args:
    :return:
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    pool = torch.nn.AvgPool2d(2)

    im2patch = utils_patch.patch_extractor(patch_size=2*args.input_height)
    example_img = utils_patch.imread('stone_imgs/img_learn_material.png').to("cpu")
    test_img = utils_patch.imread('stone_imgs/img_test_material.png').to("cpu")
    val_patches = im2patch(test_img[0].unsqueeze(0),2*args.num_samples_mmd).to(device).reshape(2*args.num_samples_mmd,1,2*args.input_height,2*args.input_height)
    val_samp = val_patches[:args.num_samples_mmd].to('cpu')
    test_samp = val_patches[args.num_samples_mmd:].to('cpu')
    val_samp_pool = pool(val_samp).to('cpu')
    test_samp_pool = pool(test_samp).to('cpu')

    val_samp = val_samp.view(args.num_samples_mmd,(2*args.input_height)**2)
    test_samp = test_samp.view(args.num_samples_mmd,(2*args.input_height)**2)


    val_samp_pool = val_samp_pool.view(args.num_samples_mmd,(args.input_height)**2)
    test_samp_pool = test_samp_pool.view(args.num_samples_mmd,(args.input_height)**2)





    logger.info('NUMBER OF PARAMETERS:')
    logger.info(sum(p.numel() for p in model.parameters()))

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("-------------------------\n")
    
    min_mmd = 1000
    min_mmd_epoch = 0.
    
    history = []
    loss_list = []
    mmd_list = []


    rev_sde.train()
    for ep in range(args.n_epochs):
        avg_loss = 0.0
        for k in range(args.steps_per_epoch):
            with torch.no_grad():
                x = im2patch(example_img[0].unsqueeze(0),args.batch_size).to(device).view(args.batch_size,1,2*args.input_height,2*args.input_height)

            with torch.no_grad():
                val_loss = rev_sde.dsm(x).mean()
                avg_loss += val_loss.item()*x.shape[0]
            if ep < args.n_epochs//2:
                loss = rev_sde.dsm(pool(x)).mean()
            else:
                loss = rev_sde.dsm(x).mean()+rev_sde.dsm(pool(x)).mean()
                
            optim.zero_grad()
            loss.backward()
            optim.step()

            

        avg_loss /= args.steps_per_epoch

        loss_list.append(avg_loss)

        if ep % args.val_freq ==0:

            with torch.no_grad():
                y0 = get_samples_batched(rev_sde, 1, 2*args.input_height, args.num_steps, args.num_samples_mmd).view(args.num_samples_mmd,(2*args.input_height)**2)
            y0 = torch.clamp(y0,0.,1.).to('cpu')
            riesz_mmd = mmd(y0, val_samp)
            history.append([ep,avg_loss,riesz_mmd])
            mmd_list.append(riesz_mmd)
            print(riesz_mmd)
            logger.info('epoch:%05d\t loss:%1.2e \t mmd:%1.2e' % (ep, avg_loss, riesz_mmd))

            if riesz_mmd < min_mmd:
                torch.save(rev_sde, args.out_dir+'/min_checkpoint.pt')
                min_mmd = riesz_mmd
                min_mmd_epoch = ep


        if ep % args.viz_freq==0:
            y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.num_samples)[0]
            y0 = torch.clamp(y0,0.,1.)

            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray')
            plt.savefig(args.out_dir+'/stone_samples_28'+str(ep))
            plt.close()
            y0 = get_samples(rev_sde, 1, 2*args.input_height, args.num_steps, args.num_samples)[0]

            y0 = torch.clamp(y0,0.,1.)

            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray')
            plt.savefig(args.out_dir+'/stone_samples_56'+str(ep))
            plt.close()

    print('MINIMUM EPOCH AND MMD')
    print(min_mmd_epoch)
    print(min_mmd)

 
    return rev_sde, history, loss_list, mmd_list, test_samp, test_samp_pool


def eval_model(rev_sde,loss_list,mmd_list,args, test_samp,test_samp_pool):
    pool = torch.nn.AvgPool2d(2)
    # plot loss curves
    plt.figure()
    plt.plot(loss_list) 
    plt.title("training loss over epochs")

    plt.savefig((args.out_dir+"/loss.png"))

    # plot eps loss
    plt.figure()
    plt.plot(mmd_list) 
    plt.title("mmd metric over epochs")

    plt.savefig((args.out_dir+"/mmd.png"))

    # save samples in folder

    rev_sde = torch.load(args.out_dir+'/min_checkpoint.pt')
    with torch.no_grad():
        y0 = get_samples_batched(rev_sde, 1, args.input_height, args.num_steps, args.num_samples_mmd).to('cpu')
    y0 = torch.clamp(y0,0.,1.).view(args.num_samples_mmd,(args.input_height)**2)
    riesz_mmd = mmd(y0, test_samp_pool)
    print("TEST MMD: ",riesz_mmd)
    with torch.no_grad():
        y0 = get_samples_batched(rev_sde, 1, 2*args.input_height, args.num_steps, args.num_samples_mmd).to('cpu')
    y0 = torch.clamp(y0,0.,1.).view(args.num_samples_mmd,(2*args.input_height)**2)
    riesz_mmd = mmd(y0, test_samp)
    print("TEST MMD: ",riesz_mmd)

    

    return 

def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior()
    elif string.lower() == "standard":
        return StandardNormal()
    elif string.lower() == "lap_conv":
        K = torch.zeros(3,3)
        hx = 1.0/args.input_height
        hy = 1.0/args.input_height
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)

        return ImplicitConv(K)
    elif string.lower() == "combined_conv":
        K = torch.zeros(3,3)
        hx = 1.0/args.input_height
        hy = 1.0/args.input_height
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)
        K = 50*K

        return CombinedConv(K)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training arguments')

    parser.add_argument('--n_epochs', type=int, default=301, help='ADAM epoch')
    parser.add_argument('--lr', type=float,default=1e-3, help='ADAM learning rate')

    parser.add_argument('--batch_size', type=int, default=64, help='number of training samples in each batch')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='number of steps per epoch')

    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--num_samples_mmd', type=int, default=20000, help='number of samples for validation')

    parser.add_argument('--num_steps', type=int, default=200, help='number of SDE timesteps')
    parser.add_argument('--input_height', type=int, default=28,  help='starting image dimensions')
    parser.add_argument('--prior_name', type=str, default='combined_conv', help="prior setup")
    
    parser.add_argument('--model', type=str, default='fno',help='nn model')
    parser.add_argument('--modes', type=int, default=8, help='cutoff modes in FNO')
    parser.add_argument('--viz_freq', type=int, default=10, help='how often to store generated images')
    parser.add_argument('--val_freq', type=int, default=10, help='validation freq')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generator')

    parser.add_argument('--out_dir', type=str, default='test', help='directory for result')
    parser.add_argument('--out_file', type=str, default='test', help='base file name for result')
    parser.add_argument('--save', type=bool, default=False,help='save from model')

    args = parser.parse_args()

    
    args.prior = choose_prior(args.prior_name)
    if args.model == "fno":
        model = FNO2d(args.modes,args.modes,64).to(device) 
    else:
        model = UNet(
        input_channels=1,
        input_height=args.input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        resamp_with_conv=True,).to(device)

    input_channels = 1

    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.out_file is not None:
        out_file = os.path.join(args.out_dir, '{:}-{:}_model_{:}_prior_{:}'.format(start_time,args.out_file,args.model,args.prior))
    else:
        out_file=None


    
    makedirs(args.out_dir)
    logger = get_logger(logpath= out_file + '.txt', filepath=os.path.abspath(__file__))
    rev_sde, history,loss_list,mmd_list, test_samp, test_samp_pool = training(args.seed, model, args,out_file=out_file)
    eval_model(rev_sde,loss_list, mmd_list,args, test_samp, test_samp_pool)

