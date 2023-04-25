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
from unet import UNet
from fno import *
from unet import *
from sde import * 
from priors import *
from utils import get_samples, save_samples, epsTest
import os
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(seed, model, args,out_file=None):
    """
    training the score function

    :param seed:
    :param model: function approximator for the score function
    :param args:
    :return:
    """

    print("seed=%d" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
    trainset = torchvision.datasets.MNIST(root='', train=True,
                                          download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)

    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)

    print('NUMBER OF PARAMETERS:')
    print(sum(p.numel() for p in model.parameters()))
    pool = torch.nn.AvgPool2d(2)

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=T )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, T, vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(rev_sde.parameters(), lr=args.lr)

    rev_sde.train()
    for ep in range(args.n_epochs):
        avg_loss = 0.0
        for k,(x,y) in enumerate(trainloader):
            x = x.to(device) 
            loss = rev_sde.dsm(pool(x)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()*x.shape[0]

        avg_loss /= len(trainset)


        if ep % args.print_freq:
            y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.batch_size)
            eps = epsTest(y0.detach(), x)
            print('EPOCH:%d\t loss:%1.2e \t eps:%1.2e' % (ep, avg_loss, eps))

        if ep % args.viz_freq:
            y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.num_samples)

            plt.figure()
            plt.imshow(torchvision.utils.make_grid(y0, 8, 5).permute((1, 2, 0)))
            plt.title("train MNIST: epoch=%d" % (ep + 1))
            if args.out_file is not None:
                plt.savefig(("%s-epoch-%d.png") % (out_file, ep + 1))
            plt.show()

    return rev_sde

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

    parser.add_argument('--batch_size', type=int, default=32, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')

    parser.add_argument('--num_steps', type=int, default=200, help='number of SDE timesteps')
    parser.add_argument('--input_height', type=int, default=16,  help='starting image dimensions')
    parser.add_argument('--prior', type=choose_prior, required=True, help="prior setup")
    
    parser.add_argument('--model', type=str, required=True,help='nn model')
    parser.add_argument('--modes', type=int, default=8, help='cutoff modes in FNO')
    parser.add_argument('--viz_freq', type=int, default=10, help='how often to store generated images')
    parser.add_argument('--print_freq', type=int, default=1, help='how often to print loss')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generator')

    parser.add_argument('--out_dir', type=str, default=None, help='directory for result')
    parser.add_argument('--out_file', type=str, default=None, help='base file name for result')

    parser.add_argument('--save', type=bool, default=False,help='save from model') 
    args = parser.parse_args()

    if args.model == "fno":
        model = FNO2d(args.modes,args.modes,64).to(device) #add u-net too
    else:
        model = UNet(
        input_channels=1,
        input_height=args.input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(256,),
        resamp_with_conv=True,).to(device)
    # starting dimensions
    input_channels = 1
    dimx = input_channels * args.input_height ** 2

    # first run
    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.out_file is not None:
        out_file = os.path.join(args.dir, '{:}-{:}_model_{:}_prior_{:}'.format(start_time,args.out_file,args.model,args.prior))
    else:
        out_file=None

    rev_sde = training(args.seed, args,out_file=out_file)
    if args.out_file is not None:
        # double check that model gets saves
        torch.save({
            'args': args,
            'rev_sde_state_dict': rev_sde.state_dict(),
            'prior_state_dict': rev_sde.prior.state_dict(),
        }, '{:}_checkpt.pth'.format(out_file))

