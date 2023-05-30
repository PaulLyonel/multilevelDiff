import torch
from pykeops.torch import LazyTensor

device = 'cpu'


    
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


def interaction_energy_term(particles_out1,r=1.,usekeops=False):
    '''
    Compute the interaction energy
    '''
    return -energy(particles_out1,particles_out1,r=r,usekeops=usekeops)

def potential_energy_term(particles_out1,target_particles,r=1.,usekeops=False):
    '''
    Compute the potential energy
    '''
    return 2*energy(particles_out1,target_particles,r=r,usekeops=usekeops)


def mmd(samples1,samples2,r = 1, use_keops= True):
    return potential_energy_term(samples1,samples2,r,use_keops)+interaction_energy_term(samples1,r,use_keops)+interaction_energy_term(samples2,r,use_keops)
