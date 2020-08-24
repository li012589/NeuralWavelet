# import encoder
import torchvision
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
#import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import os
import sys
sys.path.append(os.getcwd())

from encoder import rans, coder


def test_encodeDecode():
    #encode/decoder a MNIST image using its own distribution

    batchSize = 2
    lambd = lambda x: (x * 255).byte()
    trainsetTransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambd)
    ])
    trainSet = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=True, transform=trainsetTransform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    dataIter = iter(trainLoader)
    images, _ = dataIter.next()

    histogram = torch.histc(images.float(), bins=256, min=0, max=256)
    for i in range(len(histogram) - 1):
        histogram[i + 1] = histogram[i] + histogram[i + 1]

    CDF = [histogram[i] / histogram[len(histogram) - 1] for i in range(len(histogram))]
    states = [None for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i].reshape(-1).numpy()
        for symbol in symbols[::-1]:
            states[i] = coder.encoder(CDF, symbol, states[i])

    reconstruction = [[] for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i].reshape(-1)
        for symbol in symbols:
            states[i], recon_symbol = coder.decoder(CDF, states[i])
            if symbol != recon_symbol:
                raise ValueError
            else:
                reconstruction[i].append(recon_symbol)

    reconstruction = torch.tensor(reconstruction)
    recon_images = torch.cat([reconstruction[i].reshape(images[i].shape) for i in range(batchSize)], dim=0)

    assert_array_equal(images.reshape(batchSize, 28, 28).detach().numpy(), recon_images.detach().numpy())


def test_encodeDecode_otherDistr():
    #encode/decoder a MNIST image using OTHER distribution

    batchSize = 2
    lambd = lambda x: (x * 255).byte()
    trainsetTransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambd)
    ])
    trainSet = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=True, transform=trainsetTransform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    dataIter = iter(trainLoader)
    images, _ = dataIter.next()

    histogram = torch.randint(1, 255, [256]).to(torch.float32)

    for i in range(len(histogram) - 1):
        histogram[i + 1] = histogram[i] + histogram[i + 1]

    CDF = [histogram[i] / histogram[len(histogram) - 1] for i in range(len(histogram))]
    states = [None for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i].reshape(-1).numpy()
        for symbol in symbols[::-1]:
            states[i] = coder.encoder(CDF, symbol, states[i])

    reconstruction = [[] for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i].reshape(-1)
        for symbol in symbols:
            states[i], recon_symbol = coder.decoder(CDF, states[i])
            if symbol != recon_symbol:
                raise ValueError
            else:
                reconstruction[i].append(recon_symbol)

    reconstruction = torch.tensor(reconstruction)
    recon_images = torch.cat([reconstruction[i].reshape(images[i].shape) for i in range(batchSize)], dim=0)

    assert_array_equal(images.reshape(batchSize, 28, 28).detach().numpy(), recon_images.detach().numpy())


if __name__ == "__main__":
    test_encodeDecode_otherDistr()