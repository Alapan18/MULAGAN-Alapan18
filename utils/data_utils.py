'''import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

 
class Dataloader(Dataset):
    def __init__(self, root, dataset_name, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        print(root, dataset_name)
        self.filesA, self.filesB = self.get_file_paths(root, dataset_name)
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        img_B = Image.open(self.filesB[index % self.len])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        if dataset_name=='UIEB':
            filesA, filesB = [], []
            sub_dirs = ['train']
            for sd in sub_dirs:
                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
        return filesA, filesB 


'''
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Dataloader(Dataset):
    """ DataLoader specifically for the EUVP dataset. """
    def __init__(self, root, dataset_name, transforms_=None, mode="train"):
        """
        :param root: Path to the EUVP dataset root directory.
        :param dataset_name: Name of the dataset (used for compatibility; only EUVP is supported).
        :param transforms_: List of transformations to apply.
        :param mode: Either 'train' or 'validation'.
        """
        if dataset_name != "EUVP":
            raise ValueError("This DataLoader only supports the EUVP dataset.")
        
        self.transform = transforms.Compose(transforms_) if transforms_ is not None else None
        self.mode = mode  # 'train' or 'validation'
        self.filesA, self.filesB = self.get_file_paths(root)
        self.len = min(len(self.filesA), len(self.filesB)) if mode == "train" else len(self.filesA)

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        
        if self.mode == "train":
            img_B = Image.open(self.filesB[index % self.len])

            # Data augmentation: horizontal flip with probability 0.5
            if np.random.random() < 0.5:
                img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
                img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

            if self.transform:
                img_A = self.transform(img_A)
                img_B = self.transform(img_B)

            return {"A": img_A, "B": img_B}
        else:  # Validation mode
            if self.transform:
                img_A = self.transform(img_A)
            return {"val": img_A}

    def __len__(self):
        return self.len

    def get_file_paths(self, root):
        """
        Retrieves file paths for the EUVP dataset.
        :param root: Path to the EUVP dataset root directory.
        :return: Two lists containing file paths for domain A and domain B images.
        """
        filesA, filesB = [], []
        sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
        for sd in sub_dirs:
            filesA += sorted(glob.glob(os.path.join(root, sd, f'{self.mode}A') + "/*.*"))
            if self.mode == "train":
                filesB += sorted(glob.glob(os.path.join(root, sd, f'{self.mode}B') + "/*.*"))
        
        return filesA, filesB
