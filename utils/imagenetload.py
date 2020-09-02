from PIL import Image
import os, subprocess, pickle
import torch
import numpy as np


def unpickle(file, lineSize):
    with open(file, 'rb') as fo:
        d = pickle.load(fo)
    x = d['data']
    y = d['labels']
    y = np.array([i - 1 for i in y])
    return x, y


def load_databatch(data_folder, folders, idxs, lineSize=32):
    X = []
    Y = []
    img_size2 = lineSize * lineSize
    for i, folder in enumerate(folders):
        for no in idxs[i]:
            x, y = unpickle(os.path.join(data_folder, folder, "train_data_batch_" + str(no)), lineSize)
            x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
            x = x.reshape((x.shape[0], lineSize, lineSize, 3))
            X.append(x)
            Y.append(y)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return x, y


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, d64=False):

        if d64:
            self.trainURL = ["http://www.image-net.org/image/downsample/Imagenet64_train_part1.zip", "http://www.image-net.org/image/downsample/Imagenet64_train_part2.zip"]
            self.valURL = ["http://www.image-net.org/image/downsample/Imagenet64_val.zip"]
            folders = ["Imagenet64_train_part1", "Imagenet64_train_part2"]
            self.idxs = [range(1, 6), range(6, 11)]
            self.lineSize = 64

        else:
            self.trainURL = ["http://www.image-net.org/image/downsample/Imagenet32_train.zip"]
            self.valURL = ["http://www.image-net.org/image/downsample/Imagenet32_val.zip"]
            folders = ["Imagenet32_train"]
            self.idxs = [range(1, 11)]
            self.lineSize = 32

        super(ImageNet, self).__init__()
        self.target_transform = target_transform
        self.transform = transform

        self.train = train
        self.root = root

        if download:
            if train:
                self.download(self.trainURL)
            else:
                self.download(self.valURL)

        X, Y = load_databatch(root, folders, self.idxs, self.lineSize)
        self.data = X
        self.targets = Y

    def download(self, URL):
        folders = []
        for url in URL:
            filename = url.split('/')[-1]
            cmd = ["wget", url]
            subprocess.check_call(cmd)
            cmd = ['unzip', self.root + filename]
            subprocess.check_call(cmd)
            folders.append(filename.split(".")[0])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


if __name__ == "__main__":
    import torchvision
    lambd = lambda x: (x * 255).byte().to(torch.float32)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = ImageNet(root='./data/ImageNet32', train=True, download=False, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=500, shuffle=True)

    samples, labels = iter(targetTrainLoader).next()
    print(samples.shape)
    print(samples.max())
    print(samples.min())

    from matplotlib import pyplot as plt

    for no in range(10):
        plt.figure()
        plt.imshow(samples[no].int().permute([1, 2, 0]).numpy())
    plt.show()
