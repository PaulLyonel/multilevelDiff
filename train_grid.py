# import stuff
import argparse
import itertools
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
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.image_dir = '/local/scratch/tyang31/multilevelDiff/UTKFace' #os.getcwd() 
        self.transform = transform
        self.image_filenames = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Assuming image filenames have format 'class_###.jpg'
        label = int(self.image_filenames[idx].split('_')[0])
        
        return img, label

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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(2*args.input_height)])
    trainset = torchvision.datasets.MNIST(root='', train=True,
                                      download=True, transform=transform)

    train_set, val_set = torch.utils.data.random_split(trainset, [60000-args.num_samples_mmd, args.num_samples_mmd])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.num_samples_mmd,
                                          shuffle=True, num_workers=0)
    val_samp = next(iter(valloader))[0]

    val_samp_pool = pool(val_samp).view(args.num_samples_mmd,args.input_height**2).to('cpu')
    val_samp = val_samp.view(args.num_samples_mmd,(2*args.input_height)**2).to('cpu')




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
        for k,(x,y) in enumerate(trainloader):
            x = x.to(device) 
            loss = rev_sde.dsm(pool(x)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()*x.shape[0]

        avg_loss /= len(trainset)


        if ep % args.val_freq ==0:

            with torch.no_grad():
                y0 = get_samples_batched(rev_sde, 1, args.input_height, args.num_steps, args.num_samples_mmd).view(args.num_samples_mmd,(args.input_height)**2)
            y0 = torch.clamp(y0,0.,1.).to('cpu')
            riesz_mmd = mmd(y0, val_samp_pool)
            history.append([ep,avg_loss,riesz_mmd])
            loss_list.append(avg_loss)
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
            #plt.savefig(args.out_dir+'/mnist_samples_28ML'+str(ep))
            plt.savefig(("%s-mnist_samples_28ML-%d.png") % (out_file, ep + 1))
            plt.close()
            y0 = get_samples(rev_sde, 1, 2*args.input_height, args.num_steps, args.num_samples)[0]

            y0 = torch.clamp(y0,0.,1.)

            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray')
            plt.savefig(("%s-mnist_samples_28ML-%d.png") % (out_file, ep + 1))
            #plt.savefig(args.out_dir+'/mnist_samples_56ML'+str(ep))
            plt.close()

    print('MINIMUM EPOCH AND MMD')
    print(min_mmd_epoch)
    print(min_mmd)

 
    return rev_sde, history, loss_list, mmd_list


def eval_model(rev_sde,loss_list,mmd_list,args):
    pool = torch.nn.AvgPool2d(2)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(2*args.input_height)])

    test = torchvision.datasets.MNIST(root='', train=False,
                                          download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(test, batch_size=args.num_samples_mmd,
                                              shuffle=True, num_workers=0)
    test_samp = next(iter(testloader))[0]
    test_samp_pool = pool(test_samp).view(args.num_samples_mmd,args.input_height**2).to('cpu')

    test_samp = test_samp.view(args.num_samples_mmd,(2*args.input_height)**2).to('cpu')
    # plot loss curves
    plt.figure()
    plt.plot(loss_list) 
    plt.title("training loss over epochs")

    #plt.savefig((args.out_dir+"/loss.png"))
    plt.savefig(("%s-mnist_loss.png") % (out_file))

    # plot eps loss
    plt.figure()
    plt.plot(mmd_list) 
    plt.title("mmd metric over epochs")
    plt.savefig(("%s-mnist_mmd.png") % (out_file))
    #plt.savefig((args.out_dir+"/mmd.png"))

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

        return CombinedConv(K)
    else:
        raise argparse.ArgumentTypeError(f"Invalid class")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')

    lrvalues = [1e-3, 1e-4] 
    batches = [128, 256, 512]
    numstep_values = [100, 200]
    heights = [50, 100, 200]
    priorchoices = ['fno','combined_conv','lap_conv','standard'] 
    modelchoices = ['fno','unet']
    modeschoices = [8, 12, 14]

    combinations = list(itertools.product(lrvalues, batches, numstep_values, heights, priorchoices, modelchoices, modeschoices))

    parser.add_argument('--n_epochs', type=int, default=800, help='ADAM epoch')
    parser.add_argument('--lr', type=float,default=1e-3, choices = lrvalues, help='ADAM learning rate')

    parser.add_argument('--batch_size', type=int, default=256, choices = batches, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--num_samples_mmd', type=int, default=10000, help='number of samples for validation')

    parser.add_argument('--num_steps', type=int, default=200, choices = numstep_values, help='number of SDE timesteps')
    parser.add_argument('--input_height', type=int, default=100, choices = heights,  help='starting image dimensions')
    parser.add_argument('--prior_name', type=str, default='fno', choices = priorchoices, help="prior setup")
    
    parser.add_argument('--model', type=str, default='fno', choices = modelchoices, help='nn model')
    parser.add_argument('--modes', type=int, default=8, choices = modeschoices, help='cutoff modes in FNO')
    parser.add_argument('--viz_freq', type=int, default=10, help='how often to store generated images')
    parser.add_argument('--val_freq', type=int, default=10, help='validation freq')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generator')

    parser.add_argument('--out_dir', type=str, default='looptest', help='directory for result')
    parser.add_argument('--out_file', type=str, default='looptest', help='base file name for result')
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

    comb = 0
    for combination in combinations:
        comb = comb+1
        args.lr = combination[0]
        args.batch_size = combination[1]
        args.num_steps = combination[2]
        args.input_height = combination[3]
        args.prior_name = combination[4]
        args.model = combination[5]
        args.modes = combination[6]

        start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if args.out_file is not None:
            out_file = os.path.join(args.out_dir, '{:}-{:}_comb_{:}_model_{:}_prior_{:}'.format(start_time,args.out_file,comb,args.model,args.prior))
        else:
            out_file=None
        
        makedirs(args.out_dir)
        logger = get_logger(logpath= out_file + '.txt', filepath=os.path.abspath(__file__))
        rev_sde, history,loss_list,mmd_list = training(args.seed, model, args,out_file=out_file)
        eval_model(rev_sde,loss_list, mmd_list,args)

