import torch
import os
from PIL import Image
import logging
import numpy as np

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

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

def epsTest(X, Y, eps=1e-1):
    """
    Test for equal distributions suggested in Székely, G. J., InterStat, M. R., 2004. (n.d.).
    Testing for equal distributions in high dimension. Personal.Bgsu.Edu.

    :param X: Samples from first distribution
    :param Y: Samples from second distribution
    :param eps: conditioning paramter
    :return:
    """
    nx = X.shape[0]
    ny = Y.shape[0]

    X = X.view(nx, -1)
    Y = Y.view(ny, -1)

    sX = torch.norm(X, dim=1) ** 2
    sY = torch.norm(Y, dim=1) ** 2

    CXX = sX.unsqueeze(1) + sX.unsqueeze(0) - 2 * X @ X.t()
    CXX = torch.sqrt(CXX + eps)

    CYY = sY.unsqueeze(1) + sY.unsqueeze(0) - 2 * Y @ Y.t()
    CYY = torch.sqrt(CYY + eps)

    CXY = sX.unsqueeze(1) + sY.unsqueeze(0) - 2 * X @ Y.t()
    CXY = torch.sqrt(CXY + eps)

    D = (nx * ny) / (nx + ny) * (2.0 / (nx * ny) * torch.sum(CXY)
                                 - 1.0 / nx ** 2 * (torch.sum(CXX)) - 1.0 / ny ** 2 * (torch.sum(CYY)));

    return D / (nx + ny)
