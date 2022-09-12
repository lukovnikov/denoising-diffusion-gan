# Copied from https://github.com/sbarratt/inception-score-pytorch
import argparse
import os
import pathlib

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from tqdm import tqdm

from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy

from scripts.evaluation.fid_score import ImagePathDataset

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def inception_score(imgs, batch_size=32, resize=False, splits=1, device=torch.device("cpu")):

    assert False, "don't use this file!"

    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    print(f"Number of batches: {len(dataloader)}")
    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_is_given_path(path, batch_size, resize=32, device=torch.device("cpu")):
    """Calculates the FID of two paths"""
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])

    dataset = ImagePathDataset(files, transforms=transforms.Compose([
                                 transforms.Resize(resize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    print ("Calculating Inception Score...")
    is_mean, is_var = inception_score(dataset, batch_size=batch_size, resize=True, splits=10, device=device)
    print (is_mean, is_var)

    return is_mean, is_var


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate Inception Score')
    parser.add_argument('--path', type=str,
                        help=('Path to the generated images'))
    parser.add_argument('--batsize', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--resize', type=int, default=32,
                        help='Dimensions to resize to')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Device to use. -1 for CPU, otherwise GPU number')

    args = parser.parse_args()
    calculate_is_given_path(args.path, args.batsize, resize=args.resize,
                            device=torch.device("cpu" if args.gpu == -1 else f"cuda:{args.gpu}"))


"""
if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))

"""