from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
import random



class DDFDataset(Dataset):
    def __init__(self, source_folder, target_folder, result_folder, mask_64, mask_32, mask_16):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.result_folder = result_folder
        self.mask_64 = mask_64
        self.mask_32 = mask_32
        self.mask_16 = mask_16

        if 'val' in self.result_folder:
            self.num = len(os.listdir(self.result_folder))
            self.file_names = [f"{i}" for i in range(1, self.num + 1)]
        elif 'test' in self.result_folder:
            self.num = len(os.listdir(self.source_folder))
            self.file_names = [f"{i}" for i in range(1, self.num + 1)]

            # self.num = 100
            # self.file_names = [f"{i}" for i in range(39901, self.num + 39901)]

        else:
            # self.num = 40000
            self.num = len(os.listdir(self.source_folder))
            self.file_names = [f"{i}" for i in range(1, self.num + 1)]
            # self.file_names = [f"{i}" for i in range(20001, 60001)]
            random.shuffle(self.file_names)
        # self.file_names = [str(i) for i in random.sample(range(1, 60001), 20000)]



    def __len__(self):
        return self.num

    def __getitem__(self, index):
        while True:
            try:
                filename = self.file_names[index]
                source_name = str(filename).zfill(5) + "_source"
                base_filename = str(filename).zfill(5)
                mask_64 = os.path.join(self.mask_64, f"{base_filename}_mask_64.png")
                mask_32 = os.path.join(self.mask_32, f"{base_filename}_mask_32.png")
                mask_16 = os.path.join(self.mask_16, f"{base_filename}_mask_16.png")

                for ext in ['.jpg', '.png']:
                    source_path = os.path.join(self.source_folder, f"{base_filename}_source{ext}")
                    if os.path.exists(source_path):
                        break
                for ext in ['.jpg', '.png']:
                    target_path = os.path.join(self.target_folder, f"{base_filename}_target{ext}")
                    if os.path.exists(target_path):

                        break
                for ext in ['.jpg', '.png']:
                    result_path = os.path.join(self.result_folder, f"{base_filename}_result{ext}")
                    if os.path.exists(result_path):
                        break
                # source_path = os.path.join(self.source_folder, f"{filename}_source.jpg")
                # target_path = os.path.join(self.target_folder, f"{filename}_target.jpg")
                # result_path = os.path.join(self.result_folder, f"{filename}_result.jpg")

                source_image = Image.open(source_path).convert('RGB')
                target_image = Image.open(target_path).convert('RGB')
                result_image = Image.open(result_path).convert('RGB')

                mask_image_64 = Image.open(mask_64).convert('L')
                mask_image_32 = Image.open(mask_32).convert('L')
                mask_image_16 = Image.open(mask_16).convert('L')

                mask_image_64 = rearrange(torch.tensor(np.array(mask_image_64)).float() / 255, "h w -> 1 h w")
                mask_image_32 = rearrange(torch.tensor(np.array(mask_image_32)).float() / 255, "h w -> 1 h w")
                mask_image_16 = rearrange(torch.tensor(np.array(mask_image_16)).float() / 255, "h w -> 1 h w")

                source_image = rearrange(2 * torch.tensor(np.array(source_image)).float() / 255 - 1, "h w c -> c h w")
                target_image = rearrange(2 * torch.tensor(np.array(target_image)).float() / 255 - 1, "h w c -> c h w")
                result_image = rearrange(2 * torch.tensor(np.array(result_image)).float() / 255 - 1, "h w c -> c h w")

                return dict(edited=target_image,
                            edit=dict(c_concat=result_image, c_crossattn=source_image),
                            id=dict(source_id=source_image, target_id=target_image),
                            source_name=source_name,
                            mask=dict(mask_64=mask_image_64, mask_32=mask_image_32, mask_16=mask_image_16),
                            text="remove identity from the result face")
            except FileNotFoundError:
                index = index + 1