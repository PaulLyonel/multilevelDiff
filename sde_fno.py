import torch.nn.functional as F
import torch
import torch.nn as nn
from fno import *

device = 'cuda'
# huang sde code

def Q_g2_s(g,a):
    return g*a #g*a = Q*g^2*(gradlog p)

def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')
        
class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, alpha_min=0.1, alpha_max=20.0, T=1.0, t_epsilon=0.001, prior = None, archetype=None):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.T = T
        self.t_epsilon = t_epsilon 
        self.archetype = archetype 


    def alpha(self, t):
        return self.alpha_min + (self.alpha_max-self.alpha_min)*t

    def mean_weight(self, t): #tilde alpha
        return torch.exp(-0.25 * t**2 * (self.alpha_max-self.alpha_min) - 0.5 * t * self.alpha_min)

    def var(self, t): #tilde beta
        return 1. - torch.exp(-0.5 * t**2 * (self.alpha_max-self.alpha_min) - t * self.alpha_min)

    def f(self, t, y):
        return - 0.5 * self.alpha(t) * y

    def g(self, t, y):
        alpha_t = self.alpha(t)
        return torch.ones_like(y) * alpha_t**0.5

    def sample(self, t, y0, prior, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn(y0.shape[0],1,y0.shape[2],y0.shape[3]).to(device)

        if self.archetype == "stand":
            eta = epsilon
        if self.archetype == "lapfno":
            eta = compConv(epsilon, fun = lambda x: (1/torch.sqrt(x)))[0] 
        if self.archetype == "standfno":
            eta = prior(epsilon)

        yt = eta * std + mu
        if not return_noise:
            return yt
        else:
            return yt, eta, std, self.g(t, yt)


class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    https://github.com/CW-Huang/sdeflow-light
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False, prior = None):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
    
    # Drift
    def mu(self, t, y, prior,lmbd=0.):
        a = self.a(y, self.T - t.squeeze())
        return (1. - 0.5 * lmbd) * Q_g2_s(self.base_sde.g(self.T-t, y),a) - \
               self.base_sde.f(self.T - t, y)


    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x,prior):
        """
        denoising score matching loss
        """        
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std, g = self.base_sde.sample(t_, x, prior, return_noise=True)
        target = target.to(device) #eta

        a = self.a(y, t_.squeeze())
        score = a*std / g

        return ((score+target)**2).view(x.size(0), -1).sum(1, keepdim=False) / 2 

