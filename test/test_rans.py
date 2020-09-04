# import encoder
import torchvision
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
#import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import os
sys.path.append(os.getcwd())

from encoder import rans, coder


precision = 24

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

    nBins = 512
    mean = nBins / 8. # suppose this is the mean of the pixel distribution
    histogram = torch.histc(images.float() - mean, bins=nBins, min=-nBins//2, max=nBins//2)

    for i in range(len(histogram) - 1):
        histogram[i + 1] = histogram[i] + histogram[i + 1]

    CDF = (histogram / histogram[len(histogram) - 1]).numpy()

    # Compute CDFs, reweigh to give all bins at least
    # 1 / (2^precision) probability.
    # CDF is equal to floor[cdf * (2^precision - n_bins)] + range(n_bins)
    # To avoid divide by zero warning when using rans.
    CDFs = (CDF * ((1 << precision) - nBins)).astype('int') + np.arange(nBins)

    states = []
    # suppose all pixels have the same CDFs[0]
    CDFs = CDFs.reshape(-1, nBins)
    CDFs = np.uint32(CDFs)
    for i in range(batchSize):
        # symbols is transformed to match the indices for the CDFs array
        symbols = images[i] - mean + nBins//2
        symbols = symbols.reshape(-1).int()
        state = rans.x_init
        for j in reversed(range(len(symbols))):
            state = coder.encoder(CDFs[0], symbols[j], state)
        state = rans.flatten(state)
        states.append(state)

    reconstruction = [[] for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i] - mean + nBins//2
        symbols = symbols.reshape(-1).int()
        state = rans.unflatten(states[i])
        for symbol in symbols:
            state, recon_symbol = coder.decoder(CDFs[0], state)
            if symbol != recon_symbol:
                raise ValueError
            else:
                reconstruction[i].append(recon_symbol + mean - nBins//2)

    reconstruction = torch.tensor(reconstruction)
    recon_images = torch.cat([reconstruction[i].reshape(images[i].shape) for i in range(batchSize)], dim=0)

    assert_array_equal(images.reshape(batchSize, 28, 28).detach().numpy(), recon_images.detach().numpy())

'''
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
'''


if __name__ == "__main__":
    test_encodeDecode()