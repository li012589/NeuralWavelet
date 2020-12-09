import numpy as np
import argparse, json, math

import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt

import flow, source, train, utils


parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("Target Parameters")
group.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')

group = parser.add_argument_group("Architecture Parameters")
group.add_argument("-repeat", type=int, default=1, help="num of disentangler layers of each RG scale")
group.add_argument("-hchnl", type=int, default=12, help="intermediate channel dimension of Conv2d inside NICE inside MERA")
group.add_argument("-nhidden", type=int, default=1, help="num of intermediate channel of Conv2d inside NICE inside MERA")
group.add_argument("-nMixing", type=int, default=5, help="num of mixing distributions of last sub-priors")
group.add_argument("-simplePrior", action="store_true", help="if use simple version prior, no crossover")
group.add_argument("-clamp", type=float, default=-1, help="clamp of last prior's mean")

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
    rootFolder = './opt/default_easyMera_' + 'joinData' + "_simplePrior_" + str(args.simplePrior) + "_repeat_" + str(args.repeat) + "_hchnl_" + str(args.hchnl) + "_nhidden_" + str(args.nhidden) + "_nMixing_" + str(args.nMixing) + "_clamp_" + str(args.clamp) + "/"
    print("No specified saving path, using", rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)

if args.clamp < 0:
    args.clamp = None

# Decoding parameters to mem, saving them to save folder.
if not args.load:
    repeat = args.repeat
    hchnl = args.hchnl
    nhidden = args.nhidden
    nMixing = args.nMixing
    epoch = args.epoch
    batch = args.batch
    savePeriod = args.savePeriod
    simplePrior = args.simplePrior
    clamp = args.clamp
    lr = args.lr
    with open(rootFolder + "/parameter.json", "w") as f:
        config = {'target': 'join', 'repeat': repeat, 'hchnl': hchnl, 'nhidden': nhidden, 'nMixing': nMixing, 'epoch': epoch, 'batch': batch, 'savePeriod': savePeriod, 'lr': lr, 'simplePrior': simplePrior, 'clamp': clamp}
        json.dump(config, f)
else:
    # load saved parameters, and decoding them to mem
    with open(rootFolder + "/parameter.json", 'r') as f:
        config = json.load(f)
        locals().update(config)


lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])

trainTarget0 = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
testTarget0 = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
targetTrainLoader0 = torch.utils.data.DataLoader(trainTarget0, batch_size=batch, shuffle=True)
targetTestLoader0 = torch.utils.data.DataLoader(testTarget0, batch_size=batch, shuffle=False)

trainTarget1 = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform)
testTarget1 = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform)
targetTrainLoader1 = torch.utils.data.DataLoader(trainTarget1, batch_size=batch, shuffle=True)
targetTestLoader1 = torch.utils.data.DataLoader(testTarget1, batch_size=batch, shuffle=False)

trainTarget2 = utils.ImageNet(root='./data/ImageNet64', train=True, download=True, transform=trainsetTransform, d64=True)
testTarget2 = utils.ImageNet(root='./data/ImageNet64', train=False, download=True, transform=trainsetTransform, d64=True)
targetTrainLoader2 = torch.utils.data.DataLoader(trainTarget2, batch_size=batch, shuffle=True)
targetTestLoader2 = torch.utils.data.DataLoader(testTarget2, batch_size=batch, shuffle=False)


class JointData(object):
    def __init__(self, datas, sizes, batch):
        self.datas = datas
        self.iters = [iter(term) for term in datas]
        self.sizes = np.ceil(np.array(sizes) / batch)
        self.n = [0, 0, 0]

    def __iter__(self):
        self.iters = [iter(term) for term in self.datas]
        self.n = [0, 0, 0]
        return self

    def __next__(self):
        probs = self.sizes - np.array(self.n)
        probs = probs / np.sum(probs)
        if np.allclose(probs, np.zeros(probs.shape)):
            self.iters = [iter(term) for term in self.datas]
            self.n = [0, 0, 0]
            raise StopIteration
        no = np.argmax(np.random.multinomial(1, probs))
        self.n[no] += 1
        samples, labels = next(self.iters[no])
        print(no, samples.shape, self.n, self.sizes)
        return samples, labels


'''
class JointData(Dataset):
    def __init__(self, datas, sizes, batch):
        self.datas = [term for term in datas]
        self.sizes = np.ceil(np.array(sizes) / batch)
        self.n = [0, 0, 0]

    def __len__(self):
        return np.sum(self.sizes)

    def __getitem__(self, index):
        assert index == np.sum(self.n)
        probs = self.sizes - np.array(self.n)
        probs = probs / np.sum(probs)
        no = np.argmax(np.random.multinomial(1, probs))
        self.n[no] += 1
        samples, labels = next(self.datas[no])
        if index == self.__len__() - 1:
            self.n = [0, 0, 0]
        return samples, labels
'''


joinTargetTrainLoader = JointData([targetTrainLoader0, targetTrainLoader1, targetTrainLoader2], [len(trainTarget0), len(trainTarget1), len(trainTarget2)], batch)
joinTargetTestLoader = JointData([targetTestLoader0, targetTestLoader1, targetTestLoader2], [len(testTarget0), len(testTarget1), len(testTarget2)], batch)


targetSize = [3, 64, 64]
dimensional = 2
channel = targetSize[0]
blockLength = targetSize[-1]

decimal = flow.ScalingNshifting(256, -128)
rounding = utils.roundingWidentityGradient

depth = int(math.log(targetSize[-1], 2))


def buildLayers(shapeList):
    layers = []
    for no, chn in enumerate(shapeList[:-1]):
        if no != 0 and no != len(shapeList) - 2:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
        else:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1, padding_mode="replicate"))
        if no != len(shapeList) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    return layers


layerList = []
shapeList = [targetSize[0] * 3] + [hchnl] * (nhidden + 1) + [targetSize[0]]

for i in range(4 * repeat):
    layers = buildLayers(shapeList)
    layerList.append(torch.nn.Sequential(*layers))
    #layerList.append(torch.nn.Sequential(torch.nn.Conv2d(9, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 3, 3, padding=1)))
    torch.nn.init.zeros_(layerList[-1][-1].weight)
    torch.nn.init.zeros_(layerList[-1][-1].bias)

layerList = layerList * depth

shapeList = [targetSize[0]] + [hchnl] * (nhidden + 1) + [targetSize[0] * 3]
if not simplePrior:
    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers(shapeList)
    meanNNlist.append(torch.nn.Sequential(*layers))
    #meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 9, 3, padding=1)))
    layers = buildLayers(shapeList)
    scaleNNlist.append(torch.nn.Sequential(*layers))
    #scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 9, 3, padding=1)))
    torch.nn.init.zeros_(meanNNlist[-1][-1].weight)
    torch.nn.init.zeros_(meanNNlist[-1][-1].bias)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].weight)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].bias)

    meanNNlist = meanNNlist * depth
    scaleNNlist = scaleNNlist * depth
else:
    meanNNlist = None
    scaleNNlist = None

# Building MERA model
f = flow.SimpleMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient, clamp=clamp, compatible=True).to(device)


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

# Training
f = train.forwardKLD(f, joinTargetTrainLoader, joinTargetTestLoader, epoch, lr, savePeriod, rootFolder, plotfn=plotfn)

# Pasuse
import pdb
pdb.set_trace()
