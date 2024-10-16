import torch
import os
from PIL import Image
import logging
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# taken and modified from https://dsp.stackexchange.com/questions/93607/pixel-shift-when-using-fourier-downsampling
def downsampling_fourier(input_tensor, scaling_factor = 2):

    _ , _, height , width = input_tensor.shape  #B x C x H x W 

    # select center pixels
    center_x = height//2
    center_y = width//2
    
    
    crop_dim_x = int(center_x//scaling_factor)
    crop_dim_y = int(center_y//scaling_factor)

    fimage = torch.fft.fftshift(torch.fft.fft2(input_tensor, norm = "ortho"))

    fft_crop = fimage[:,:,(center_x-crop_dim_x):(center_x+crop_dim_x),(center_y-crop_dim_y):(center_y+crop_dim_y)]

    
    tensor_downsampled = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_crop), norm = "ortho"))
    

    return tensor_downsampled



def sample_test(sde, input_channels, input_height, num_steps, num_samples):
    """

    generates samples from the reverse SDE

    :param sde: instance of SDE class
    :param input_channels:
    :param input_height: resolution of input images
    :param num_steps: number time steps for sampling
    :param num_samples: number of samples
    :return:
    """

    delta = sde.T / num_steps
    y0 = sde.prior.sample([num_samples, input_channels, input_height, input_height])
    y1 = downsampling_fourier(y0)
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, lmbd = 1.)
            sigma = sde.sigma(ones * ts[i], y0, lmbd = 1.)
            epsilon = sde.prior.sample(y0.shape)
            y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y1, lmbd = 1.)
            sigma = sde.sigma(ones * ts[i], y1, lmbd = 1.)
            epsilon = sde.prior.sample(y1.shape)
            y1 = y1 + delta * mu + (delta ** 0.5) * sigma * epsilon
    return y0, y1

def closed_form_score2(sde,matrix, prior, t, x):
    res = x.shape[2]
    matrix = matrix.to(device)#+1*torch.eye(matrix.shape[0], device = device)#.unsqueeze(0).repeat(x.shape[0],1,1)
    matrix_inv = torch.linalg.inv(matrix)


    
    
    x = x.requires_grad_()
    var_weight = sde.base_sde.var_weight(t) 
    mean_weight = sde.base_sde.mean_weight(t)
    
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
  
    prod1 = prod1
    prod2 = prod2

    vector1 = -0.5*((prod1**2).sum(dim = 1))
    vector2 = -0.5*((prod2**2).sum(dim = 1))

    score =  torch.logsumexp(torch.cat((vector1.unsqueeze(0), vector2.unsqueeze(0)),0), dim = 0).view(x.shape[0])
    grad = torch.autograd.grad(score.sum(), x)[0].view(x.shape[0],res**2)
    Q = matrix@matrix.T
    return (grad@Q).view(x.shape[0],1,res,res)*sde.base_sde.g(t, x)**2-sde.base_sde.f(t,x)



def get_samples_true(sde, input_channels, input_height, num_steps, num_samples, matrix, prior):
    """

    generates samples from the reverse SDE

    :param sde: instance of SDE class
    :param input_channels:
    :param input_height: resolution of input images
    :param num_steps: number time steps for sampling
    :param num_samples: number of samples
    :return:
    """

    delta = sde.T / num_steps
    y0 = sde.prior.sample([num_samples, input_channels, input_height, input_height])
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    for i in range(num_steps):
        mu = closed_form_score2(sde, matrix, prior, sde.T-ts[i],y0).detach()
        sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
        epsilon = sde.prior.sample(y0.shape)
        y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon

    return y0

def get_samples(sde, input_channels, input_height, num_steps, num_samples, store_itermediates=True):
    """

    generates samples from the reverse SDE

    :param sde: instance of SDE class
    :param input_channels:
    :param input_height: resolution of input images
    :param num_steps: number time steps for sampling
    :param num_samples: number of samples
    :return:
    """

    delta = sde.T / num_steps
    y0 = sde.prior.sample([num_samples, input_channels, input_height, input_height])
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    Y = []

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, lmbd = 0.)
            sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
            epsilon = sde.prior.sample(y0.shape)
            y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon
            if store_itermediates:
                Y.append(y0)
    return y0, Y

def get_samples_batched(sde, input_channels, input_height, num_steps, num_samples):
    """

    generates samples from the reverse SDE

    :param sde: instance of SDE class
    :param input_channels:
    :param input_height: resolution of input images
    :param num_steps: number time steps for sampling
    :param num_samples: number of samples
    :return:
    """

    delta = sde.T / num_steps
   
    samples = torch.empty(0, device = device)
    for l in range(100):
        y0 = sde.prior.sample([num_samples//100, input_channels, input_height, input_height])
        ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
        ones = torch.ones(num_samples//100, 1, 1, 1).to(y0)

        with torch.no_grad():
            for i in range(num_steps):
                mu = sde.mu(ones*ts[i], y0, lmbd = 0.)
                sigma = sde.sigma(ones*ts[i], y0, lmbd = 0.)
                epsilon = sde.prior.sample(y0.shape)
                y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon

        samples = torch.cat((samples, y0),0)

    return samples


def save_samples(y0, i,folder):
    """

    save samples as individual jpg images

    :param y0: generated images
    :param file_name: base file name (without the exension)
    :return:
    """
    makedirs(str(folder))

    for j in range(y0.shape[0]):
        y0j = torch.clamp(y0[j], 0., 1.)
        arr = y0j.cpu().data.numpy() * 255
        arr = arr.astype(np.uint8).squeeze(0)

        im = Image.fromarray(arr)
        
        print(str(folder+str(i*100+j)+'.jpeg'))
        im.save(str(folder+str(i*100+j)+'.jpeg'))
