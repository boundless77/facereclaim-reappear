import argparse
import cv2
import numpy as np
import torch
import os
from config import Config
from models import resnet_face18
from models.fs_networks import Generator_Adain_Upsample
import torch.nn.parallel
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import random
from einops import rearrange
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic multi-scale attribute-mask extraction for ANY face-swap method "
                    "(based on attribute/injection/extract_attribute_*.py)")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="folder of the result faces (files named <id>_result.jpg / <id>_result.png, "
                             "id zero-padded to 5 digits)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="output folder; mask_64 / mask_32 / mask_16 sub-folders are created inside")
    parser.add_argument("--gnet_weights", type=str, default="weights/G_net_epoch_1000.pth",
                        help="attribute encoder (Generator_Adain_Upsample) weights")
    parser.add_argument("--face_weights", type=str, required=True,
                        help="resnet_face18 identity model weights (the original config's test_model_path)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    return parser.parse_args()


args = parse_args()
device_ids = [args.gpu]


def fix_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # "module."
        new_state_dict[new_key] = value
    return new_state_dict


def get_attmap1(input=None):
    outputs = (input ** 0.5).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)

    am = (outputs[0, ...]).cpu().detach().numpy()
    # am = cv2.resize(am, (256, 256))
    am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
    am = np.uint8(np.floor(am))
    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
    am = cv2.cvtColor(am, cv2.COLOR_BGR2RGB)
    return am


def get_attmap_oppt(input=None):
    outputs = (input ** 2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)

    am = (outputs[0, ...]).cpu().detach().numpy()
    am = cv2.resize(am, (256, 256))
    am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
    am = np.uint8(np.floor(am))
    am = 255-am
    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
    am = cv2.cvtColor(am, cv2.COLOR_BGR2RGB)

    plt.imshow(am)
    plt.show()


def get_attmap2(input=None):
    output = torch.mean(input, dim=1)
    output_tensor = torch.squeeze(output).cpu()
    output_numpy = output_tensor.detach().numpy()

    min_val, max_val = np.min(output_numpy), np.max(output_numpy)
    output_numpy = (output_numpy - min_val) / (max_val - min_val)


    quantized_output = np.zeros_like(output_numpy)


    quantized_output[output_numpy == 0] = 0
    quantized_output[(output_numpy > 0) & (output_numpy <= 0.2)] = 0.2
    quantized_output[(output_numpy > 0.2) & (output_numpy <= 0.4)] = 0.4
    quantized_output[(output_numpy > 0.4) & (output_numpy <= 0.6)] = 0.6
    quantized_output[(output_numpy > 0.6) & (output_numpy <= 0.8)] = 0.8
    quantized_output[(output_numpy > 0.8) & (output_numpy <= 1.0)] = 1.0
    quantized_output = (quantized_output*255).astype(np.uint8)

    output_numpy = cv2.resize(quantized_output, (256, 256), interpolation=cv2.INTER_LINEAR)

    am = cv2.applyColorMap(output_numpy, cv2.COLORMAP_JET)
    am = cv2.cvtColor(am, cv2.COLOR_BGR2RGB)
    plt.imshow(am)
    plt.show()

    # plt.imshow(output_numpy, cmap="jet")
    # plt.colorbar()
    # plt.show()


class AttDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.num = len(os.listdir(self.img_path))
        self.file_names = [f"{i}" for i in range(1, self.num + 1)]
        # random.shuffle(self.file_names)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        while True:
            try:
                filename = self.file_names[index]
                base_filename = str(filename).zfill(5)
                jpg_path = os.path.join(self.img_path, f"{base_filename}_result.jpg")
                png_path = os.path.join(self.img_path, f"{base_filename}_result.png")
                if os.path.exists(jpg_path):
                    img_path = jpg_path
                elif os.path.exists(png_path):
                    img_path = png_path
                else:
                    raise FileNotFoundError(f"File not found: {jpg_path} or {png_path}")

                # Load color image using cv2
                color_image = cv2.imread(img_path)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image = rearrange(2 * torch.tensor(color_image).float() / 255 - 1, "h w c -> c h w")

                # Load grayscale image using cv2
                grayscale_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                grayscale_image = cv2.resize(grayscale_image, (128, 128))
                grayscale_image = 2 * torch.tensor(grayscale_image).float() / 255 - 1
                grayscale_image = grayscale_image.unsqueeze(0)  # Add channel dimension
                return color_image, grayscale_image, base_filename

            except FileNotFoundError:
                index = index + 1


  # ----  ----
os.makedirs(os.path.join(args.mask_dir, 'mask_64'), exist_ok=True)
os.makedirs(os.path.join(args.mask_dir, 'mask_32'), exist_ok=True)
os.makedirs(os.path.join(args.mask_dir, 'mask_16'), exist_ok=True)

test_dataset = AttDataset(img_path=args.result_dir)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

att_encoder = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9)
att_encoder = nn.DataParallel(att_encoder, device_ids=device_ids)
att_weights = torch.load(args.gnet_weights)
att_encoder.load_state_dict(att_weights)

opt = Config()
model = resnet_face18(opt.use_se)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load(args.face_weights))
model.eval()
model.requires_grad_(False)


net_G = att_encoder.to(device_ids[0])
model = model.to(device_ids[0])


def negative_correlation_loss(att_features, id_features):
    loss = 0
    correlation = 0
    for f_att, f_id in zip(att_features, id_features):
  # (batch_size, feature_size)
        f_att = f_att.view(f_att.size(0), -1)
        f_id = f_id.view(f_id.size(0), -1)


        f_att_mean = f_att.mean(dim=1, keepdim=True)
        f_id_mean = f_id.mean(dim=1, keepdim=True)


        f_att_centered = f_att - f_att_mean
        f_id_centered = f_id - f_id_mean


        numerator = torch.sum(f_att_centered * f_id_centered, dim=1)
        denominator = torch.norm(f_att_centered, p=2, dim=1) * torch.norm(f_id_centered, p=2, dim=1)


        correlation = numerator / (denominator + 1e-8)
        loss += (1 + correlation).mean()
    return correlation


for batch_idx, (color_image, grayscale_image, base_filename) in tqdm(enumerate(test_dataloader)):

    # output_path = f'...mask_64/{base_filename[0]}_mask_64.png'
    # if os.path.exists(output_path):
    #     continue

    color_image = color_image.to(device_ids[0])
    grayscale_image = grayscale_image.to(device_ids[0])

    id, _ = model(grayscale_image)

    _, att_features = net_G(color_image, id)

    att_input64 = att_features[0]
    att_input32 = att_features[1]
    att_input16 = att_features[2]

    mask_64 = get_attmap1(att_input64)
    mask_32 = get_attmap1(att_input32)
    mask_16 = get_attmap1(att_input16)

    plt.imsave(os.path.join(args.mask_dir, 'mask_64', f'{base_filename[0]}_mask_64.png'), mask_64)
    plt.imsave(os.path.join(args.mask_dir, 'mask_32', f'{base_filename[0]}_mask_32.png'), mask_32)
    plt.imsave(os.path.join(args.mask_dir, 'mask_16', f'{base_filename[0]}_mask_16.png'), mask_16)

print(len(os.listdir(os.path.join(args.mask_dir, 'mask_64'))))
print(len(os.listdir(os.path.join(args.mask_dir, 'mask_32'))))
print(len(os.listdir(os.path.join(args.mask_dir, 'mask_16'))))

    # get_attmap_oppt(id_input64)
    # get_attmap_oppt(id_input32)
    # get_attmap_oppt(id_input16)

    # get_attmap1(att_input64)
    # get_attmap1(att_input32)
    # get_attmap1(att_input16)
