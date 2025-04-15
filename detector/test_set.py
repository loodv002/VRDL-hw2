import torch

import os
from PIL import Image

class TestSet(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.image_names = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = f'{self.root}/{self.image_names[idx]}'
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return self.image_names[idx], image

    def __len__(self):
        return len(self.image_names)