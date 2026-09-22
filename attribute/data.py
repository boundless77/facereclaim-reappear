from __future__ import annotations
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
import random


class AttDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.num = len(os.listdir(self.img_path))
        self.file_names = [f"{i}" for i in range(1, self.num + 1)]
        random.shuffle(self.file_names)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        filename = self.file_names[index]
        base_filename = str(filename).zfill(5)
        img_path = os.path.join(self.img_path, f"{base_filename}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")

        return image

if __name__ == '__main__':
    dataset = AttDataset(img_path='/path/to/your/attribute/')
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[0].dtype)
    print(dataset[0].max())
    print(dataset[0].min())
