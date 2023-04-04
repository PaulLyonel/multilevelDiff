import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from fno import *

device = 'cuda'
#archetecture
model = FNO2d(8,8,64).to(device)
#add UNet later

#stencil for cov operator 
#Laplacian
def stencil(hx, hy):
    K = torch.zeros(3,3)
    K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
    K[0,1] = -1.0/(hy**2)
    K[1,0] = -1.0/(hx**2)
    K[2,1] = -1.0/(hy**2)
    K[1,2] = -1.0/(hx**2)

    return K

def compConv(x,fun= lambda x : x):
    """
    compute fun(conv_op(K))*x assuming periodic boundary conditions on x

    where
    K       - is a 2D convolution stencil (assumed to be separable)
    conv_op - means that we build the matrix representation of the operator
    fun     - is a function applied to the operator (as a matrix-function, not
component-wise), default fun(x)=x
    x       - are images, torch.tensor, shape=N x 1 x nx x ny
    """
    n = x.shape
    #stencil
    nx = n[2]
    ny = n[3]
    hx = 1.0/nx
    hy = 1.0/ny
    K = stencil(hx, hy).to(device)
    m = K.shape
    mid1 = (m[0]-1)//2
    mid2 = (m[1]-1)//2
    Bp = torch.zeros(n[2],n[3], device = device)
    Bp[0:mid1+1,0:mid2+1] = K[mid1:,mid2:]
    Bp[-mid1:, 0:mid2 + 1] = K[0:mid1, -(mid2 + 1):]
    Bp[0:mid1 + 1, -mid2:] = K[-(mid1 + 1):, 0:mid2]
    Bp[-mid1:, -mid2:] = K[0:mid1, 0:mid2]
    xh = torch.fft.rfft2(x)
    Bh = torch.fft.rfft2(Bp)
    lam = fun(torch.abs(Bh)).to(device)
    xh = xh.to(device)
    lam[torch.isinf(lam)] = 0.0
    xBh = xh * lam.unsqueeze(0).unsqueeze(0)
    xB = torch.fft.irfft2(xBh)
    xB = 9*xB

    return xB,lam

############specify prior######

# def prior(x):
#     return x

#trace class covariance operator
# def prior(x):
#     return compConv(x, fun = lambda x: (1/torch.sqrt(x)))[0]

#truncated prior
def prior(x):
    return SpectralConv2d(1,1,16,9, rand = False).to(device)(x)

###############

def prior_distr(shape_vec, prior):
    epsilon = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
    eta = prior(epsilon)
    return eta

##loss##
#Q*g^2*(gradlog p)
def Q_g2_s(g,a): 
    return g*a #prior(g*a) 

