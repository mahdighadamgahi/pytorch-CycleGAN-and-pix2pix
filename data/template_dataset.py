"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""

import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import torch
from torchvision import transforms  # Import transforms from torchvision

class TemplateDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets from .pt files.

    It requires two .pt files to host training images from domain A and domain B respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.file_A = os.path.join(opt.dataroot, opt.phase + 'A.pt')  # path to the .pt file for domain A
        self.file_B = os.path.join(opt.dataroot, opt.phase + 'B.pt')  # path to the .pt file for domain B

        self.A_data = torch.load(self.file_A,weights_only=False)  # load the dataset for domain A
        self.B_data = torch.load(self.file_B,weights_only=False)  # load the dataset for domain B

        self.A_size = len(self.A_data)  # get the size of dataset A
        self.B_size = len(self.B_data)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths (optional if needed)
            B_paths (str)    -- image paths (optional if needed)
        """
        A_img, _ = self.A_data[index % self.A_size]  # get image from dataset A
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_img, _ = self.B_data[index_B]  # get image from dataset B

        # Convert tensors to PIL images
        A_img = transforms.ToPILImage()(A_img)
        B_img = transforms.ToPILImage()(B_img)

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': '', 'B_paths': ''}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
