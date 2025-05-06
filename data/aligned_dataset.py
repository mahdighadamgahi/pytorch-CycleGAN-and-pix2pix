import os
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from torchvision.transforms import ToPILImage
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset stored in a .pt file.

    The .pt file should contain a tensor dataset where each entry corresponds to an image.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_path = os.path.join(opt.dataroot, opt.phase + '.pt')  # Path to the .pt file
        self.dataset = torch.load(self.data_path,weights_only=False)  # Load the dataset from the .pt file
        self.to_pil = ToPILImage()  # Initialize ToPILImage transformation
        assert(self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain (left half)
            B (tensor) - - its corresponding image in the target domain (right half)
            A_paths (str) - - dummy path for domain A
            B_paths (str) - - dummy path for domain B
        """
        # Load the image tensor from the dataset
        image_tensor, _ = self.dataset[index]  # Assuming each entry is (image_tensor, label)

        # Convert tensor to a PIL image
        AB = self.to_pil(image_tensor)

        # Split the image into A (left half) and B (right half)
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # Apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': f'index_{index}_A', 'B_paths': f'index_{index}_B'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)
