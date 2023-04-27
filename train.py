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
from utils import get_samples, save_samples, epsTest, makedirs, get_logger
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

    logger.info("seed=%d" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
    trainset = torchvision.datasets.MNIST(root='', train=True,
                                          download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)

    T = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)

    logger.info('NUMBER OF PARAMETERS:')
    logger.info(sum(p.numel() for p in model.parameters()))
    pool = torch.nn.AvgPool2d(2)

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=T )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, T, vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(rev_sde.parameters(), lr=args.lr)

    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("-------------------------\n")

    history = []


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


        if (ep+1) % args.print_freq:
            y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.batch_size)[0]
            eps = epsTest(y0.detach(), pool(x))
            history.append([ep,avg_loss,eps.item()])
            logger.info('epoch:%05d\t loss:%1.2e \t eps:%1.2e' % (ep, avg_loss, eps))

        if (ep+1) % args.viz_freq:
            y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.num_samples)[0]
            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=8, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray') #for 3d: 
            plt.title("train MNIST: epoch=%d" % (ep + 1))
            if out_file is not None:
                plt.savefig(("%s-epoch-%d.png") % (out_file, ep + 1))
            plt.show()

    return rev_sde, history

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
    parser.add_argument('--prior', type=choose_prior, default='fno', help="prior setup")
    
    parser.add_argument('--model', type=str, default='fno',help='nn model')
    parser.add_argument('--modes', type=int, default=8, help='cutoff modes in FNO')
    parser.add_argument('--viz_freq', type=int, default=1, help='how often to store generated images')
    parser.add_argument('--print_freq', type=int, default=1, help='how often to print loss')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generator')

    parser.add_argument('--out_dir', type=str, default='test', help='directory for result')
    parser.add_argument('--out_file', type=str, default='test', help='base file name for result')

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

    input_channels = 1

    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.out_file is not None:
        out_file = os.path.join(args.out_dir, '{:}-{:}_model_{:}_prior_{:}'.format(start_time,args.out_file,args.model,args.prior))
    else:
        out_file=None


    
    makedirs(args.out_dir)
    logger = get_logger(logpath= out_file + '.txt', filepath=os.path.abspath(__file__))
    logger.info("start time: " + start_time)
    logger.info(args)



    rev_sde, history = training(args.seed, model, args,out_file=out_file)
    


    if out_file is not None:
        # double check that model gets saves
        torch.save({
            'args': args,
            'history': history,
            'rev_sde_state_dict': rev_sde.state_dict(),
            'prior_state_dict': rev_sde.prior.state_dict(),
        }, '{:}_checkpt.pth'.format(out_file))

