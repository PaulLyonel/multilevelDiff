import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from fno import *



class StandardNormal():

    # def __init__(self, shape_vec):
    #     self.shape_vec = shape_vec

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

class FNOprior():
    def __init__(self):
        self.conv = SpectralConv2d(1,1,16,9, rand = False).to(device)

    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    def Qmv(self,v):
        return self.conv(v)

    def Q_g2_s(self, g,a): 
        return g*a #Qmv(g*a) 


class ImplicitConv():
    # N(0,Q), Q is a separable convolution with periodic boundary condition

    # def __init__(self, shape_vec):
    #     self.shape_vec = shape_vec

    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    def Qmv(self,v):
        return self.compConv(v, fun = lambda x: (1/torch.sqrt(x)))[0]

    def Q_g2_s(self, g,a): #method
        return g*a #self.Qmv(g*a) 

    #Laplacian
    def stencil(self, hx, hy):
        K = torch.zeros(3,3)
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)

        return K

    def compConv(self,x,fun= lambda x : x):
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
        K = self.stencil(hx, hy).to(device)
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


class testQConv():
    # def __init__(self, shape_vec):
    #     shape_vec = shape_vec

    def sample(self,shape_vec):
        xlength = shape_vec[2]*shape_vec[3]
        x = torch.eye(xlength).to(device) #identity cov for standard normal
        Q = self.compConv1(x,fun= lambda x : 1/x)[0] #cov
        mvn = td.multivariate_normal.MultivariateNormal(loc=torch.zeros(xlength).to(device),covariance_matrix=Q)
        epsilon = mvn.sample_n(shape_vec[0])
        eta = torch.reshape(epsilon,[shape_vec[0],1,shape_vec[2],shape_vec[3]])
        return eta
    
    def Qmv(self, v):
        return v

    def Q_g2_s(self, g,a): 
        return g*a #Qmv(g*a) 

    #Laplacian
    def stencil(self, hx, hy):
        K = torch.zeros(3,3)
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)

        return K

    def compConv1(self, x,fun= lambda x : x):
        """
        compute fun(conv_op(K))*x assuming periodic boundary conditions on x
        where
        K       - is a 2D convolution stencil (assumed to be separable)
        conv_op - means that we build the matrix representation of the operator
        fun     - is a function applied to the operator (as a matrix-function, not
    component-wise), default fun(x)=x
        x       - are covariance operator, torch.tensor, dim = nx^2 x ny^2
        """
        n = x.shape
        #stencil
        nx = n[0]
        ny = n[1]
        hx = 1.0/nx
        hy = 1.0/ny
        K = self.stencil(hx, hy).to(device)
        m = K.shape
        mid1 = (m[0]-1)//2
        mid2 = (m[1]-1)//2
        Bp = torch.zeros(n[0],n[1], device = device)
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



