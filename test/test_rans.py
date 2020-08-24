# import encoder
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from encoder import rans, coder


def test_encodeDecode():
    #encode/decoder a MNIST image using its own distribution

    batchSize = 5
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

    def imshow(img, figure):
        npImg = img.numpy()
        plt.imshow(np.transpose(npImg, (1, 2, 0)))
        figure.show()

    fig1 = plt.figure(1)
    imshow(torchvision.utils.make_grid(images), fig1)

    histogram = torch.histc(images.float(), bins=256, min=0, max=256)
    for i in range(len(histogram) - 1):
        histogram[i + 1] = histogram[i] + histogram[i + 1]

    CDF = [histogram[i] / histogram[len(histogram) - 1] for i in range(len(histogram))]
    states = [None for i in range(batchSize)]
    for i in range(batchSize):
        print("Encoding "+str(i+1)+"-th image")
        symbols = images[i].reshape(-1).numpy()
        for symbol in symbols[::-1]:
            states[i] = coder.encoder(CDF, symbol, states[i])

    print("Encoding finished.")

    reconstruction = [[] for i in range(batchSize)]
    for i in range(batchSize):
        print("Decoding " + str(i+1) + "-th image")
        symbols = images[i].reshape(-1)
        for symbol in symbols:
            states[i], recon_symbol = coder.decoder(CDF, states[i])
            if symbol != recon_symbol:
                raise ValueError
            else:
                reconstruction[i].append(recon_symbol)

    reconstruction = torch.tensor(reconstruction)
    recon_images = [reconstruction[i].reshape(images[i].shape) for i in range(batchSize)]
    fig2 = plt.figure(2)
    imshow(torchvision.utils.make_grid(recon_images), fig2)

    print("Decoding finished.")
    input()



def test_encodeDecode_otherDistr():
    #encode/decoder a MNIST image using OTHER distribution
    pass


if __name__ == "__main__":
    test_encodeDecode()