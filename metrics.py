import torch
from pykeops.torch import LazyTensor
import scipy
import scipy.linalg
from numpy import linalg
import numpy as np

device = 'cpu'

# https://github.com/layer6ai-labs/dgm-eval
def entropy_q(p, q=1):
    p_ = p[p > 0]
    return -(p_ * np.log(p_)).sum()


# taken from caterini et al exposing flaws diffusion eval
# https://github.com/layer6ai-labs/dgm-eval
def sw_approx(X, Y):
    '''Approximate Sliced W2 without
    Monte Carlo From https://arxiv.org/pdf/2106.15427.pdf'''
    d = X.shape[1]
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    mean_term = linalg.norm(mean_X - mean_Y) ** 2 / d
    m2_Xc = (linalg.norm(X - mean_X, axis=1) ** 2).mean() / d
    m2_Yc = (linalg.norm(Y - mean_Y, axis=1) ** 2).mean() / d
    approx_sw = (mean_term + (m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2)) ** 2) ** (1/2)
    return approx_sw

# taken from caterini et al exposing flaws diffusion eval 
# https://github.com/layer6ai-labs/dgm-eval
def compute_vendi_score(X, q=1):
    X = X/(np.sqrt(np.sum(X**2, axis = 1))[:,None])
    n = X.shape[0]
    S = X @ X.T
    w = scipy.linalg.eigvalsh(S / n)
    return np.exp(entropy_q(w, q=q))


def differences_keops(p,q):
    '''
    Compute the pairwise differences using the module pykeops
    '''
    q_k = LazyTensor(q.unsqueeze(1).contiguous())
    p_k = LazyTensor(p.unsqueeze(0).contiguous())
    rmv = q_k-p_k
    return rmv

def differences(p,q):
    '''
    Compute the pairwise differences
    '''
    dim = p.shape[1]
    m_p, m_q = p.shape[0], q.shape[0]
    diff = p.reshape(m_p,1,dim) - q.reshape(1,m_q,dim)
    return diff

def distance(p,q,diff=None,usekeops=False):
    '''
    Compute the norms of the pairwise differences
    '''
    if usekeops:
        if diff is None:
            diff = differences_keops(p,q) + 1e-13
        out = (diff**2).sum(2).sqrt()
    else:
        if diff is None:
            diff = differences(p,q)
        out=torch.linalg.vector_norm(diff,ord=2,dim=2)
    return out

def energy(p,q,r=1.,usekeops=False):
    '''
    Sum up over all computed distances
    '''
    dist = distance(p,q,usekeops=usekeops)
    if usekeops:
        return 0.5*((dist**r).sum(0).sum(0))/(p.shape[0]*q.shape[0])
    else:
        return 0.5*torch.sum(dist**r)/(p.shape[0]*q.shape[0])


def interaction_energy_term(particles_out1,r=1.,usekeops=True):
    '''
    Compute the interaction energy
    '''
    return -energy(particles_out1,particles_out1,r=r,usekeops=usekeops)

def potential_energy_term(particles_out1,target_particles,r=1.,usekeops=True):
    '''
    Compute the potential energy
    '''
    return 2*energy(particles_out1,target_particles,r=r,usekeops=usekeops)


def mmd(samples1,samples2,r = 1, use_keops= True):
    return potential_energy_term(samples1,samples2,r,use_keops)+interaction_energy_term(samples1,r,use_keops)+interaction_energy_term(samples2,r,use_keops)
