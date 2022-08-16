import os
import pickle
from typing import Optional, Callable, Any, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

import numpy as np
from torchvision.transforms import transforms


class AFHQDataset(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    # base_folder = "cifar-10-batches-py"
    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # tgz_md5 = "c58f30108f718f92721af3b95e74349a"

    # meta = {
    #     "filename": "batches.meta",
    #     "key": "label_names",
    #     "md5": "5ff9c542aee3614f3951f8cda6e48888",
    # }

    def __init__(
        self,
        root: str,
        subset:str="cat",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.split = "train" if self.train is True else "test"
        self.subset = subset

        self.filenames = []
        for name in os.listdir(os.path.join(self.root, self.split, self.subset)):
            if os.path.isfile(os.path.join(self.root, self.split, self.subset, name)):
                self.filenames.append(name)

        # if download:
        #     self.download()
        #
        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.split, self.subset, self.filenames[index]))

        target = torch.tensor([0])

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self) -> int:
        return len(self.filenames)

    # def _check_integrity(self) -> bool:
    #     root = self.root
    #     for fentry in self.train_list + self.test_list:
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    # def download(self) -> None:
    #     if self._check_integrity():
    #         print("Files already downloaded and verified")
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


if __name__ == '__main__':
    ds = AFHQDataset("./datasets/afhq/", "cat", transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    print(len(ds))
    x = ds[5][0]
    print(x.size())