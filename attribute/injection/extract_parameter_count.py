
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


load_weights = True

dataset = AttDataset(img_path='/path/to/your/attribute/')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# test_dataset = AttDataset(img_path='/path/to/your/FaceReclaim/test-2500/result/result_uniface-new')
test_dataset = AttDataset(img_path='/path/to/your/attribute/result_zzw/result/')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device_ids = [0]

att_encoder = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9)
att_encoder = nn.DataParallel(att_encoder, device_ids=device_ids)
if load_weights:
    att_weights = torch.load("/path/to/your/attribute/injection/weights/G_net_epoch_1000.pth")
    fixed_att_weights = fix_state_dict_keys(att_weights)
    if len(device_ids) > 1:
        att_encoder.load_state_dict(att_weights)
    else:
        att_encoder.load_state_dict(att_weights)

opt = Config()
model = resnet_face18(opt.use_se)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load(opt.test_model_path))
model.eval()
model.requires_grad_(False)


net_G = att_encoder.to(device_ids[0])
model = model.to(device_ids[0])


def unwrap_dp(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net

def count_params(net, trainable_only=False):
    net = unwrap_dp(net)
    if trainable_only:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in net.parameters())

def fmt(n):
    return f"{n/1e6:.3f}M"


id_branch = model          # resnet_face18
att_branch = net_G         # Generator_Adain_Upsample

id_total = count_params(id_branch, trainable_only=False)
id_train = count_params(id_branch, trainable_only=True)

att_total = count_params(att_branch, trainable_only=False)
att_train = count_params(att_branch, trainable_only=True)

fad_total = id_total + att_total
fad_train = id_train + att_train

print("===== Parameter Count =====")
print(f"ID branch    : total {fmt(id_total)}   | trainable {fmt(id_train)}")
print(f"Attr branch  : total {fmt(att_total)}  | trainable {fmt(att_train)}")
print(f"FAD (sum)    : total {fmt(fad_total)}  | trainable {fmt(fad_train)}")
