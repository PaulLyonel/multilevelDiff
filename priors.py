import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from fno import *




class StandardNormal(nn.Module):

    def __init__(self):
        super(StandardNormal, self).__init__()

    def __repr__(self):
        return "StandardNormal"

    #standard noise
    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    #covariance
    def Qmv(self,v):
        return v

    #loss structure
    #g*a means the Q in reverse drift is learned implicitly in a
    #self.Qmv(g*a) means the Q is given explictly
    def Q_g2_s(self, g,a): 
        return g*a 

class FNOprior(nn.Module):
    def __init__(self,k1=28,k2=14, scale = 1.):
        super(FNOprior, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.conv = SpectralConv2d(1,1,k1,k2, rand = False).to(device)
        self.scale = scale

    def __repr__(self):
        return "FNOprior(k1=%d, k2=%d)" %(self.k1,self.k2)
    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.scale*self.Qmv(x)

    def Qmv(self,v):
        return self.conv(v)

    def Q_g2_s(self, g,a): 
        return g*a #Qmv(g*a) 



class BesselConv(nn.Module):
    def __init__(self, scale=6, power=0.55):
        super(BesselConv, self).__init__()
        self.scale = scale
        self.power = power


    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    def Qmv(self,v):
        return self.compConv(v, fun = lambda x: (1/(x**self.power)))[0] 


    def Q_g2_s(self, g,a): 
        return g*a 


    def compConv(self, x, fun=lambda x: x):
        """
        Compute fun(scale*I - conv_op(K))*x assuming periodic boundary conditions on x

        Parameters:
        x       - Input images, torch.tensor, shape=N x 1 x nx x ny
        fun     - A function applied to the operator (as a matrix-function, not component-wise), default fun(x)=x
        """
    
        n = x.shape
        K = torch.zeros(3, 3, device=x.device)
        hx = 1.0 / x.shape[2]
        hy = 1.0 / x.shape[3]
        K[1, 1] = 2.0 / (hx**2) + 2.0 / (hy**2)
        K[0, 1] = -1.0 / (hy**2)
        K[1, 0] = -1.0 / (hx**2)
        K[2, 1] = -1.0 / (hy**2)
        K[1, 2] = -1.0 / (hx**2)
        
        m = K.shape
        mid1 = (m[0] - 1) // 2
        mid2 = (m[1] - 1) // 2
        Bp = torch.zeros(n[2], n[3], device=x.device)
        Bp[0:mid1+1, 0:mid2+1] = K[mid1:, mid2:]
        Bp[-mid1:, 0:mid2+1] = K[0:mid1, -(mid2 + 1):]
        Bp[0:mid1+1, -mid2:] = K[-(mid1 + 1):, 0:mid2]
        Bp[-mid1:, -mid2:] = K[0:mid1, 0:mid2]

        xh = torch.fft.rfft2(x)
        Bh = torch.fft.rfft2(Bp)

        scale_identity = self.scale * torch.ones_like(Bh)
        scale_Bh = scale_identity + torch.abs(Bh)

        lam = fun(scale_Bh).to(x.device)
        lam[torch.isinf(lam)] = 0.0

        xh = xh.to(x.device)
        xBh = xh * lam.unsqueeze(0).unsqueeze(0)

        xB = torch.fft.irfft2(xBh, s=(x.shape[2], x.shape[3]))
        return xB, lam
    
class ImplicitConv(nn.Module):
    """
    Compute the covariance matrix as

    Q = conv(K)^{-1/2}

    where K is a  stencil, e.g., the standard 5-point Laplacian.
    """

    def __init__(self, scale=10):
        super(ImplicitConv, self).__init__()
        self.scale = scale

    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    def Qmv(self,v):
        return self.compConv(v, fun = lambda x: (1/x**0.55))[0]

    def Q_g2_s(self, g,a): #method
        return g*a #self.Qmv(g*a) 



    def compConv(self,x,fun= lambda x : x):
        """
        compute fun(conv_op(K))*x assuming periodic boundary conditions on x

        where
        x       - are images, torch.tensor, shape=N x 1 x nx x ny
        fun     - is a function applied to the operator (as a matrix-function, not
    component-wise), default fun(x)=x
        """

        n = x.shape
        K = torch.zeros(3,3)
        hx = 1.0/x.shape[2]
        hy = 1.0/x.shape[3]
        K[1,1] = 2.0/(hx**2)+ 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)
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
        xB = self.scale * torch.fft.irfft2(xBh)
        return xB,lam
    
class CombinedConv(nn.Module):
    """
    Compute the covariance matrix as

    Q = conv(K)^{-1/2}

    where K is a  stencil, e.g., the standard 5-point Laplacian.
    """

    def __init__(self,k1=28,k2=14,scale=10,scale2 = 1.):
        super(CombinedConv, self).__init__()
        self.scale = scale
        self.scale2 = scale2
        self.conv = SpectralConv2d(1,1,k1,k2, rand = False).to(device)


    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    def Qmv(self,v):
        return self.scale2*self.conv(v)+self.compConv(v, fun = lambda x: (1/x**0.55))[0]

    def Q_g2_s(self, g,a): 
        return g*a 



    def compConv(self,x,fun= lambda x : x):
        """
        compute fun(conv_op(K))*x assuming periodic boundary conditions on x

        where
        x       - are images, torch.tensor, shape=N x 1 x nx x ny
        fun     - is a function applied to the operator (as a matrix-function, not
    component-wise), default fun(x)=x
        """

        n = x.shape
        K = torch.zeros(3,3)
        hx = 1.0/x.shape[2]
        hy = 1.0/x.shape[3]
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)
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
        xB = (self.scale)*torch.fft.irfft2(xBh)
        return xB,lam



