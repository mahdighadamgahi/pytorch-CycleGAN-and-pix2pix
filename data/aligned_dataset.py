import os
import torch
from data.base_dataset import BaseDataset, get_params, get_transform


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset stored in a .pt file.

    The .pt file should contain a list of tuples, where each tuple contains:
    - A (PIL image or tensor): the image in the input domain
    - B (PIL image or tensor): the image in the target domain
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_path = os.path.join(opt.dataroot, opt.phase + '.pt')  # path to the .pt file
        self.data = torch.load(self.data_path,weights_only=False)  # load the data from the .pt file

        assert self.opt.load_size >= self.opt.crop_size, \
            "Crop size should be smaller than or equal to the load size."
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - a dummy path for domain A
            B_paths (str) - - a dummy path for domain B
        """
        # Retrieve paired data (A, B)
        A, B = self.data[index]

        # Apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': f'index_{index}_A', 'B_paths': f'index_{index}_B'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
