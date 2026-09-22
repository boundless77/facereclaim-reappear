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
# class DDFDataset(Dataset):
#     def __init__(self, source_folder, target_folder, result_folder):
#         self.source_folder = source_folder
#         self.target_folder = target_folder
#         self.result_folder = result_folder
#
#         if 'val' in self.result_folder:
#             self.num = len(os.listdir(result_folder))
#         elif 'test' in self.result_folder:
#             self.num = 500
#         else:
#             self.num = 60000
#
#         # self.file_names = [f"{i}" for i in range(1, self.num + 1)]
#         # self.file_names = [f for f in self.file_names if os.path.exists(os.path.join(self.source_folder, f"{f}_source.jpg")) or os.path.exists(os.path.join(self.source_folder, f"{f}_source.png"))]
#         # random.shuffle(self.file_names)
#         # self.num = len(self.file_names)  # set self.num to the actual length of self.file_names
#
#         self.file_names = [f"{i}" for i in range(1, self.num + 1)]
#         if 'val' or 'test' not in self.result_folder:
#             random.shuffle(self.file_names)
#
#     def __len__(self):
#         return self.num
#
#     def __getitem__(self, index):
#         while True:
#             try:
#                 filename = self.file_names[index]
#                 source_name = str(filename).zfill(5) + "_source"
#                 base_filename = str(filename).zfill(5)
#                 for ext in ['.jpg', '.png']:
#                     source_path = os.path.join(self.source_folder, f"{base_filename}_source{ext}")
#                     if os.path.exists(source_path):
#                         break
#                 for ext in ['.jpg', '.png']:
#                     target_path = os.path.join(self.target_folder, f"{base_filename}_target{ext}")
#                     if os.path.exists(target_path):
#                         break
#                 for ext in ['.jpg', '.png']:
#                     result_path = os.path.join(self.result_folder, f"{base_filename}_result{ext}")
#                     if os.path.exists(result_path):
#                         break
#                 # source_path = os.path.join(self.source_folder, f"{filename}_source.jpg")
#                 # target_path = os.path.join(self.target_folder, f"{filename}_target.jpg")
#                 # result_path = os.path.join(self.result_folder, f"{filename}_result.jpg")
#
#                 source_image = Image.open(source_path).convert('RGB')
#                 target_image = Image.open(target_path).convert('RGB')
#                 result_image = Image.open(result_path).convert('RGB')
#
#                 source_image = rearrange(2 * torch.tensor(np.array(source_image)).float() / 255 - 1, "h w c -> c h w")
#                 target_image = rearrange(2 * torch.tensor(np.array(target_image)).float() / 255 - 1, "h w c -> c h w")
#                 result_image = rearrange(2 * torch.tensor(np.array(result_image)).float() / 255 - 1, "h w c -> c h w")
#
#                 return dict(edited=target_image,
#                             edit=dict(c_concat=result_image, c_crossattn=source_image),
#                             id=dict(source_id=source_image, target_id=target_image),
#                             source_name=source_name)
#             except FileNotFoundError:
#                 index = index + 1


class DDFDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.num = len(os.listdir(self.img_path))
        self.file_names = [f"{i}" for i in range(1, self.num + 1)]
        random.shuffle(self.file_names)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
            filename = self.file_names[index]
            source_name = str(filename).zfill(5) + "_source"
            base_filename = str(filename).zfill(5)

            img_path = os.path.join(self.img_path, f"{base_filename}_source.jpg")
            source_image = Image.open(img_path).convert('RGB')

            source_image = rearrange(2 * torch.tensor(np.array(source_image)).float() / 255 - 1, "h w c -> c h w")

            return source_image
