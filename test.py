# import stuff
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from fno import *
from unet_no_att import *
from sde import * 
from priors import *
from utils import get_samples, makedirs, get_logger, get_samples_batched
import os
import datetime
from metrics import *


parser = argparse.ArgumentParser(description='Test arguments')
parser.add_argument('--dataset', type=str, default='MNIST', choices = ['MNIST', 'FashionMNIST'], help='dataset types')
parser.add_argument('--out_dir', type=str, default='test', help='directory for result')
parser.add_argument('--out_file', type=str, default='test', help='base file name for result')
parser.add_argument('--save_model', type=str, default=False,help='saved checkpoint')
parser.add_argument('--upscale', type=int, default=2,help='multiscale level')
parser.add_argument('--input_height', type=int, default=28,  help='starting image dimensions')
parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
parser.add_argument('--num_samples_mmd', type=int, default=10000, help='number of samples for test')
parser.add_argument('--num_steps', type=int, default=200, help='number of SDE timesteps')
args = parser.parse_args()


start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if args.out_file is not None:
    out_file = os.path.join(args.out_dir, '{:}-{:}'.format(start_time,args.out_file))
else:
    out_file=None

makedirs(args.out_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = get_logger(logpath= out_file + '.txt', filepath=os.path.abspath(__file__))


logger.info(args.save_model)
rev_sde = torch.load(args.save_model + '.pt')
pool = torch.nn.AvgPool2d(args.upscale)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.upscale*args.input_height)])

# Download and load the test data
if args.dataset == 'MNIST':
    testset = datasets.MNIST(root='', train=False, download=False, transform=transform)
elif args.dataset == 'FashionMNIST':
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=False, train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.num_samples_mmd, shuffle=True, num_workers=1)
test_samp = next(iter(testloader))[0]
test_samp_pool = pool(test_samp).view(args.num_samples_mmd,args.input_height**2).to('cpu')
test_samp = test_samp.view(args.num_samples_mmd,(args.upscale*args.input_height)**2).to('cpu')


with torch.no_grad():
    y0 = get_samples_batched(rev_sde, 1, args.input_height, args.num_steps, args.num_samples_mmd).to('cpu')
y0 = torch.clamp(y0,0.,1.).view(args.num_samples_mmd,(args.input_height)**2)
diversity = compute_vendi_score(y0)
riesz_mmd = mmd(y0, test_samp_pool)


with torch.no_grad():
    y0 = get_samples_batched(rev_sde, 1, 2*args.input_height, args.num_steps, args.num_samples_mmd).to('cpu')
y0 = torch.clamp(y0,0.,1.).view(args.num_samples_mmd,(args.upscale*args.input_height)**2)
diversityhigh = compute_vendi_score(y0)
riesz_mmdhigh = mmd(y0, test_samp)


logger.info('diversity:%1.4e \t riesz_mmd:%1.4e \t diversityhigh:%1.4e \t riesz_mmdhigh: :%1.4e' % (diversity,  riesz_mmd, diversityhigh, riesz_mmdhigh))

y0 = get_samples(rev_sde, 1, args.input_height, args.num_steps, args.num_samples)[0]
y0 = torch.clamp(y0,0.,1.)

plt.figure()
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

plt.imshow(image_grid.cpu().numpy(), cmap='gray')
plt.savefig(args.out_dir+'/mnist_samples_28ML'+args.save_model)
plt.close()
y0 = get_samples(rev_sde, 1, args.upscale*args.input_height, args.num_steps, args.num_samples)[0]

y0 = torch.clamp(y0,0.,1.)

plt.figure()
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))
plt.imshow(image_grid.cpu().numpy(), cmap='gray')
plt.savefig(args.out_dir+'/mnist_samples_56*2ML'+args.save_model)
plt.close()
