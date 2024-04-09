from torch.utils.data import Dataset
import torch
import numpy as np
import gzip
import pathlib
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import nibabel as nib

class GANDataset(Dataset):
    def __init__(self, directory_path):
        print(directory_path)
        self.directory_path = pathlib.Path(directory_path)
        IMAGE_EXTENSIONS = ['nii', 'nii.gz']
        self.files = sorted([file for ext in IMAGE_EXTENSIONS
                             for file in self.directory_path.glob('*.{}'.format(ext))])

    def __getitem__(self, index):

        image_path = self.files[index]

        img = nib.load(image_path)
        data = img.get_fdata()

        image_normalized = self.__intensity_normalize_one_volume__(data)
        data = torch.FloatTensor(image_normalized).unsqueeze(0)

        return data

    def __len__(self):
        return len(self.files)

    def __intensity_normalize_one_volume__(self, volume):
        """
        Normalize the intensity of an nd volume based on the mean and std of non-zero region.
        For image data, consider whether you need this step as-is, or adjust based on your needs.
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]

        return out  # normalized_volume
