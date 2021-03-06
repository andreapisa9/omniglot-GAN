import os
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from torchvision.utils import save_image
from torchvision import datasets, transforms

from PIL import Image


class customOmniglot(datasets.Omniglot):
    def __init__(self, root, label, background=True, transform=None, target_transform=None,
                 download=False, mask_mode = False):
        super(customOmniglot, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.label = label
        self.train = background  # training set or test set

        # if True create loader with N - 1 classes
        self.mask_mode = mask_mode

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.label_data = []
        self.label_target = []

        if self.mask_mode == False:
            # Load only one label
            for i, (d,l) in enumerate(zip(self.data,self.targets)):
                if(int(l) == self.label):
                    self.label_data.append(d)
                    self.label_target.append(l)
            
            print("LabelOmniglot {}".format(self.label))
        elif self.mask_mode == True:
            # Load N -1 labels
            for i, (d,l) in enumerate(zip(self.data,self.targets)):
                if(int(l) != self.label):
                    self.label_data.append(d)
                    self.label_target.append(l)

            print("LabelOmniglot masked {}".format(self.label))

       

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target,mean_pixel) where target is index of the target class.
        """
        
        img, target = self.label_data[index], int(self.label_target[index])
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        
        return img, target

    def __len__(self):
        return len(self.label_data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'Omniglot', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'Omniglot', 'processed')
