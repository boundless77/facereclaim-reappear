import os
import numpy as np
from PIL import Image


mask_64_dir = '/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_64'
mask_32_dir = '/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_32'
mask_16_dir = '/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_16'


os.makedirs(mask_64_dir, exist_ok=True)
os.makedirs(mask_32_dir, exist_ok=True)
os.makedirs(mask_16_dir, exist_ok=True)


white_image_64 = np.ones((64, 64, 3), dtype=np.uint8) * 255
white_image_32 = np.ones((32, 32, 3), dtype=np.uint8) * 255
white_image_16 = np.ones((16, 16, 3), dtype=np.uint8) * 255

for i in range(1, 60001):
    filename_64 = os.path.join(mask_64_dir, f'{i:05d}_mask_64.png')
    filename_32 = os.path.join(mask_32_dir, f'{i:05d}_mask_32.png')
    filename_16 = os.path.join(mask_16_dir, f'{i:05d}_mask_16.png')

    if not os.path.exists(filename_64):
        Image.fromarray(white_image_64).save(filename_64)
    if not os.path.exists(filename_32):
        Image.fromarray(white_image_32).save(filename_32)
    if not os.path.exists(filename_16):
        Image.fromarray(white_image_16).save(filename_16)