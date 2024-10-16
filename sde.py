import torch.nn.functional as F
import torch
import torch.nn as nn
from priors import *
from fno import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# this code is based on the implementation https://github.com/CW-Huang/sdeflow-light
# please cite their paper if you use this code 

class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, prior, alpha_min=0.1, alpha_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.prior = prior
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.T = T
        self.t_epsilon = t_epsilon 
        
    def alpha(self, t):
        return self.alpha_min + (self.alpha_max-self.alpha_min)*t

    def mean_weight(self, t): #tilde alpha in paper
        return torch.exp(-0.25 * t**2 * (self.alpha_max-self.alpha_min) - 0.5 * t * self.alpha_min)

    def var_weight(self, t): #tilde beta in paper
        return 1. - torch.exp(-0.5 * t**2 * (self.alpha_max-self.alpha_min) - t * self.alpha_min)

    def f(self, t, y):
        return - 0.5 * self.alpha(t)*y

    def g(self, t, y):
        alpha_t = self.alpha(t)
        return torch.ones_like(y) * alpha_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t)*y0  #mean = tilde alpha * y0
        std_weight = self.var_weight(t)**0.5        
        eta = self.prior.sample(y0.shape) 
        yt = eta * std_weight + mu
        if not return_noise:
            return yt
        else:
            return yt, eta, std_weight, self.g(t, yt)


class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    https://github.com/CW-Huang/sdeflow-light
    """

    def __init__(self, prior,base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.prior = prior
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
    
    # Drift
    def mu(self, t, y, lmbd=0.):
        a = self.a(y, self.T - t.squeeze())
        return (1. - 0.5 * lmbd) * self.prior.Q_g2_s(self.base_sde.g(self.T-t, y),a) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)
    
    def dsm(self, x):
        """
        denoising score matching loss
        """ 
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std_weight, g = self.base_sde.sample(t_, x, return_noise=True)
        target = target.to(device) #eta

        a = self.a(y, t_.squeeze())
        score = a*std_weight/ g  #a =(Q*(y-mu)*g)/(std_weight*Q)

        fin = score+target
        return ((fin)**2).view(x.size(0), -1).sum(1, keepdim=False) / 2 

