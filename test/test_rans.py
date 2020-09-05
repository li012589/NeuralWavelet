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
torch.manual_seed(42)

def test_encodeDecode():
    #encode/decoder a MNIST image using its own distribution

    batchSize = 10
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
    images = images.float()

    nBins = 512

    HISTos = []
    MEANs = []
    theoryBPD = []
    CDFs = []

    for i in range(batchSize):
        import pdb
        pdb.set_trace()
        mean = int(images[i].mean().item())
        MEANs.append(mean)
        histogram = torch.histc(images[i] - mean, bins=nBins, min=-nBins // 2, max=nBins // 2)
        HISTos.append(histogram)

        prob = histogram / np.prod(images[i].shape)
        logp = -torch.log2(prob)
        logp[logp == float('inf')] = 0
        theorySize = (logp * histogram).sum().item()
        theoryBPD.append(theorySize / np.prod(images.shape[1:]))

        cdf = prob.numpy()
        for j in range(prob.shape[0] - 1):
            cdf[j + 1] = cdf[j] + cdf[j + 1]
        CDFs.append(torch.from_numpy(((cdf * ((1 << precision) - nBins)).astype('int') + np.arange(nBins)).reshape(1, nBins)))

    MEANs = torch.tensor(MEANs).reshape(batchSize, 1)
    CDFs = torch.cat(CDFs, 0).numpy()

    print("theory BPD:", np.mean(theoryBPD))

    states = []
    # suppose all pixels have the same CDFs[0]
    CDFs = np.uint32(CDFs)
    for i in range(batchSize):
        # symbols is transformed to match the indices for the CDFs array
        symbols = images[i] - MEANs[i] + nBins // 2
        symbols = symbols.reshape(-1).int()
        state = rans.x_init
        for j in reversed(range(len(symbols))):
            state = coder.encoder(CDFs[i], symbols[j], state)
        state = rans.flatten(state)
        states.append(state)

    print("actual BPD:", np.mean([32 * len(term) / np.prod(images.shape[1:]) for term in states]))
    import pdb
    pdb.set_trace()

    reconstruction = [[] for i in range(batchSize)]
    for i in range(batchSize):
        symbols = images[i] - MEANs[i] + nBins//2
        symbols = symbols.reshape(-1).int()
        state = rans.unflatten(states[i])
        for symbol in symbols:
            state, recon_symbol = coder.decoder(CDFs[i], state)
            if symbol != recon_symbol:
                raise ValueError
            else:
                reconstruction[i].append(recon_symbol + MEANs[i] - nBins//2)

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