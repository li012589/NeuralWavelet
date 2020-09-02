import numpy as np
import argparse, json, math

import torch, torchvision
from torch import nn
import matplotlib.pyplot as plt

import flow, source, train, utils


parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("Target Parameters")
group.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet', 'MNIST'], metavar='DATASET', help='Dataset choice.')

group = parser.add_argument_group("Architecture Parameters")
group.add_argument("-depth", type=int, default=-1, help="depth of hierarchy structure, -1 means full depth")
group.add_argument("-repeat", type=int, default=1, help="num of disentangler layers of each RG scale")
group.add_argument("-nhidden", type=int, default=3, help="num of MLP layers inside NICE inside MERA")
group.add_argument("-hdim", type=int, default=50, help="layer dimension of MLP inside NICE inside MERA")
group.add_argument("-nNICE", type=int, default=2, help="num of NICE layers of each RG scale (even number only)")
group.add_argument("-nMixing", type=int, default=5, help="num of mixing distributions of last sub-priors")
group.add_argument("-smallPrior", action='store_true', help="use a smaller prior to save params")

group = parser.add_argument_group('Learning  parameters')
group.add_argument("-epoch", type=int, default=400, help="num of epoches to train")
group.add_argument("-batch", type=int, default=200, help="batch size")
group.add_argument("-savePeriod", type=int, default=10, help="save after how many steps")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")

group = parser.add_argument_group("Etc")
group.add_argument("-folder", default=None, help="Path to save")
group.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
group.add_argument("-load", action='store_true', help="If load or not")

args = parser.parse_args()

device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))

# Creating save folder
if args.folder is None:
    rootFolder = './opt/default_' + args.target + "_depth_" + str(args.depth) + "_repeat_" + str(args.repeat) + "_nhidden_" + str(args.nhidden) + "_hdim_" + str(args.hdim) + "_Sprior_" + str(args.smallPrior) + "/"
    print("No specified saving path, using", rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)

# Decoding parameters to mem, saving them to save folder.
if not args.load:
    target = args.target
    depth = args.depth
    repeat = args.repeat
    nhidden = args.nhidden
    hdim = args.hdim
    nNICE = args.nNICE
    nMixing = args.nMixing
    smallPrior = args.smallPrior
    epoch = args.epoch
    batch = args.batch
    savePeriod = args.savePeriod
    lr = args.lr
    with open(rootFolder + "/parameter.json", "w") as f:
        config = {'target': target, 'depth': depth, 'repeat': repeat, 'nhidden': nhidden, 'hdim': hdim, 'nNICE': nNICE, 'nMixing': nMixing, 'smallPrior': smallPrior, 'epoch': epoch, 'batch': batch, 'savePeriod': savePeriod, 'lr': lr}
        json.dump(config, f)
else:
    # load saved parameters, and decoding them to mem
    with open(rootFolder + "/parameter.json", 'r') as f:
        config = json.load(f)
        locals().update(config)

# Building the target dataset
if target == "CIFAR":
    # Define dimensions
    targetSize = [3, 32, 32]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
    testTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)
elif args.target == "ImageNet32":
    # Define dimensions
    targetSize = [3, 32, 32]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform)
    testTarget = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

elif args.target == "ImageNet64":
    # Define dimensions
    targetSize = [3, 64, 64]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform, d64=True)
    testTarget = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform, d64=True)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

elif args.target == "MNIST":
    pass
else:
    raise Exception("No such target")

# Building the sub-priors
# TODO: depth less than int(math.log(blockLength, 2))
priorList = []
_length = int((blockLength * blockLength) / 4)
for n in range(int(math.log(blockLength, 2))):
    if n != (int(math.log(blockLength, 2))) - 1:
        # intermedia variable prior, 3 here means the left 3 variable
        if smallPrior:
            priorList.append(source.DiscreteLogistic([channel, 1, 3], decimal, rounding)
        else:
            priorList.append(source.DiscreteLogistic([channel, _length, 3], decimal, rounding)
    elif n == depth - 1:
        # if depth is specified, the last prior
        priorList.append(source.MixtureDiscreteLogistic([channel, _length, 4], nMixing, decimal, rounding)
        break
    else:
        # final variable prior, all 4 variable
        priorList.append(source.MixtureDiscreteLogistic([channel, _length, 4], nMixing, decimal, rounding)
    _length = int(_length / 4)

# Building the hierarchy prior
p = source.HierarchyPrior(channel, blockLength, priorList, repeat=repeat)


# Building NICE model inside MERA
assert nNICE % 2 == 0
assert depth <= int(math.log(blockLength, 2))

layerList = []
if depth == -1:
    depth = None
# NOTE HERE: Same wavelet at each RG scale. If want different wavelet, change (repeat + 1)
#            to depth * (repeat + 1)!
for _ in range(repeat + 1):
    maskList = []
    for n in range(nNICE):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0)#.to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 2 * 1] + [hdim] * nhidden + [3 * 2 * 1], [nn.ELU()] * nhidden + [None]) for _ in range(nNICE)]
    layerList.append(flow.DiscreteNICE(maskList, tList, decimal, rounding))

# Building MERA model
f = flow.MERA(dimensional, blockLength, layerList, repeat, depth=depth, prior=p).to(device)

# sanity check of f and it's prior
for no in range(int(math.log(blockLength, 2))):
    if no == depth:
        # early break if depth is specified
        break
    if no != int(math.log(blockLength, 2)) - 1:
        np.testing.assert_allclose(f.indexI[(no + 1) * (repeat + 1) - 1][:, 1:], f.prior.factorOutIList[no])
    else:
        np.testing.assert_allclose(f.indexI[(no + 1) * (repeat + 1) - 1], f.prior.factorOutIList[no])

# Define plot function
def plotfn(f, train, test, LOSS, VALLOSS):
    # loss plot
    lossfig = plt.figure(figsize=(8, 5))
    lossax = lossfig.add_subplot(111)

    epoch = len(LOSS)
    lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
    lossax.plot(np.arange(epoch), np.array(VALLOSS), 'ro-', label="val. loss", markersize=2.5)

    lossax.set_xlim(0, epoch)
    lossax.legend()
    lossax.set_title("Loss Curve")
    plt.savefig(rootFolder + 'pic/lossCurve.png', bbox_inches="tight", pad_inches=0)
    plt.close()

    # wavelet plot, Fig. 3 in draft

    # draw samples, same samples
    samplesNum = 10
    samples, _ = iter(train).next()
    samples = samples[:samplesNum].to(device)

    # build a shallow flow, change var _depth here wouldn't change how plot behave
    _depth = 2
    ftest = flow.MERA(dimensional, blockLength, f.layerList[:(_depth * (repeat + 1))], repeat, depth=_depth).to(device)

    # do the transformations
    z, _ = ftest.inverse(samples)

    # collect parts
    zparts = []
    for no in range(_depth):
        _, z_ = utils.dispatch(p.factorOutIList[no], p.factorOutJList[no], z)
        zparts.append(z_)

    # the inner upper left part
    _, zremain = utils.dispatch(ftest.indexI[-1], ftest.indexJ[-1], z)
    _linesize = np.sqrt(zremain.shape[-2]).astype(np.int)
    zremain = zremain[:, :, :, :1].reshape(*zremain.shape[:-2], _linesize, _linesize)

    # define renorm fn
    def back01(tensor):
        ten = tensor.clone()
        ten = ten.view(ten.shape[0], -1)
        ten -= ten.min(1, keepdim=True)[0]
        ten /= ten.max(1, keepdim=True)[0]
        ten = ten.view(tensor.shape)
        return ten

    # norm the remain
    zremain = back01(zremain)

    for i in range(_depth):

        # inner parts, order: upper right, down left, down right
        parts = []
        for no in range(3):
            part = back01(zparts[-(i + 1)][:, :, :, no].reshape(*zremain.shape))
            parts.append(part)

        # piece the inner up
        zremain = torch.cat([zremain, parts[0]], dim=-1)
        tmp = torch.cat([parts[1], parts[2]], dim=-1)
        zremain = torch.cat([zremain, tmp], dim=-2)

    # convert zremaoin to numpy array
    zremain = zremain.permute([0, 2, 3, 1]).detach().cpu().numpy()

    # convert samples to numpy int array
    samples = samples.to(torch.int).permute([0, 2, 3, 1]).detach().cpu().numpy()

    for no in range(samplesNum):
        waveletPlot = plt.figure(figsize=(8, 8))
        waveletAx = waveletPlot.add_subplot(111)
        waveletAx.imshow(zremain[no])
        plt.savefig(rootFolder + 'pic/waveletPlot' + str(no) + '.png', bbox_inches="tight", pad_inches=0)
        plt.close()
        originalPlot = plt.figure(figsize=(8, 8))
        originalAx = originalPlot.add_subplot(111)
        originalAx.imshow(samples[no])
        plt.savefig(rootFolder + 'pic/originalPlot' + str(no) + '.png', bbox_inches="tight", pad_inches=0)
        plt.close()

# Training
f = train.forwardKLD(f, targetTrainLoader, targetTestLoader, epoch, lr, savePeriod, rootFolder, device, plotfn=plotfn)

# Pasuse
import pdb
pdb.set_trace()
