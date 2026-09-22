
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


def mi_and_nmi_2d(a, b, bins=64, eps=1e-12, use_log2=False):
    """
    a,b: (H,W) float arrays, aligned
    return: MI, NMI
    """
    assert a.ndim == 2 and b.ndim == 2, "MI expects 2D scalar maps (H,W), not RGB heatmaps."

    a = a.reshape(-1)
    b = b.reshape(-1)

    # normalize to [0,1] for fixed binning
    a = (a - a.min()) / (a.max() - a.min() + eps)
    b = (b - b.min()) / (b.max() - b.min() + eps)

    ai = np.clip((a * bins).astype(np.int64), 0, bins - 1)
    bi = np.clip((b * bins).astype(np.int64), 0, bins - 1)

    joint = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1.0)
    joint = joint / (joint.sum() + eps)

    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)

    log = np.log2 if use_log2 else np.log

    mi = np.sum(joint * (log(joint + eps) - log(pa + eps) - log(pb + eps)))

    Ha = -np.sum(pa * log(pa + eps))
    Hb = -np.sum(pb * log(pb + eps))
    nmi = (2.0 * mi) / (Ha + Hb + eps)

    return float(mi), float(nmi)

def feat_to_scalar_map(feat: torch.Tensor, eps=1e-12):
    """
    feat: (B, C, H, W)  torch tensor
    return: (H, W) numpy float32, BEFORE colormap
    """
  # abs + sqrt feat  sqrt  NaN
    m = torch.sqrt(torch.abs(feat) + eps).sum(dim=1)  # (B, H, W)

    b, h, w = m.shape
    m = m.view(b, -1)
    m = F.normalize(m, p=2, dim=1).view(b, h, w)      # per-image L2 norm
    return m[0].detach().cpu().numpy().astype(np.float32)


mi64_list, mi32_list, mi16_list = [], [], []
nmi64_list, nmi32_list, nmi16_list = [], [], []


for batch_idx, (color_image, grayscale_image, base_filename) in tqdm(enumerate(test_dataloader)):

    # output_path = f'/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_64/{base_filename[0]}_mask_64.png'
    # if os.path.exists(output_path):
    #     continue

    color_image = color_image.to(device_ids[0])
    grayscale_image = grayscale_image.to(device_ids[0])

    id, id_features = model(grayscale_image)

    _, att_features = net_G(color_image, id)
    
    # id_input64 = id_features[0]
    # id_input32 = id_features[1]
    # id_input16 = id_features[2]

    # id_mask_64 = get_attmap1(id_input64)
    # id_mask_32 = get_attmap1(id_input32)
    # id_mask_16 = get_attmap1(id_input16)


    # att_input64 = att_features[0]
    # att_input32 = att_features[1]
    # att_input16 = att_features[2]


    # att_mask_64 = get_attmap1(att_input64)
    # att_mask_32 = get_attmap1(att_input32)
    # att_mask_16 = get_attmap1(att_input16)

    id_map_64  = feat_to_scalar_map(id_features[0])
    id_map_32  = feat_to_scalar_map(id_features[1])
    id_map_16  = feat_to_scalar_map(id_features[2])

    att_map_64 = feat_to_scalar_map(att_features[0])
    att_map_32 = feat_to_scalar_map(att_features[1])
    att_map_16 = feat_to_scalar_map(att_features[2])

    mi64, nmi64 = mi_and_nmi_2d(id_map_64,  att_map_64, bins=64)
    mi32, nmi32 = mi_and_nmi_2d(id_map_32,  att_map_32, bins=64)
    mi16, nmi16 = mi_and_nmi_2d(id_map_16,  att_map_16, bins=64)

    mi64_list.append(mi64); nmi64_list.append(nmi64)
    mi32_list.append(mi32); nmi32_list.append(nmi32)
    mi16_list.append(mi16); nmi16_list.append(nmi16)


    # plt.imsave(f'/path/to/your/FaceReclaim/test-2500/mask_FAD/uniface-new/mask_64/{base_filename[0]}_mask_64.png', mask_64)
    # plt.imsave(f'/path/to/your/FaceReclaim/test-2500/mask_FAD/uniface-new/mask_32/{base_filename[0]}_mask_32.png', mask_32)
    # plt.imsave(f'/path/to/your/FaceReclaim/test-2500/mask_FAD/uniface-new/mask_16/{base_filename[0]}_mask_16.png', mask_16)
    # plt.imsave(f'result_zzw/id/mask_64/{base_filename[0]}_mask_64.png', id_mask_64)
    # plt.imsave(f'result_zzw/id/mask_32/{base_filename[0]}_mask_32.png', id_mask_32)
    # plt.imsave(f'result_zzw/id/mask_16/{base_filename[0]}_mask_16.png', id_mask_16)
    # plt.imsave(f'result_zzw/att/mask_64/{base_filename[0]}_mask_64.png', att_mask_64)
    # plt.imsave(f'result_zzw/att/mask_32/{base_filename[0]}_mask_32.png', att_mask_32)
    # plt.imsave(f'result_zzw/att/mask_16/{base_filename[0]}_mask_16.png', att_mask_16)


def mean_std(x):
    x = np.array(x, dtype=np.float64)
    return x.mean(), x.std()

mi64_m, mi64_s = mean_std(mi64_list)
mi32_m, mi32_s = mean_std(mi32_list)
mi16_m, mi16_s = mean_std(mi16_list)

nmi64_m, nmi64_s = mean_std(nmi64_list)
nmi32_m, nmi32_s = mean_std(nmi32_list)
nmi16_m, nmi16_s = mean_std(nmi16_list)

print(f"MI@64 : {mi64_m:.4f} ± {mi64_s:.4f} | NMI@64 : {nmi64_m:.4f} ± {nmi64_s:.4f}")
print(f"MI@32 : {mi32_m:.4f} ± {mi32_s:.4f} | NMI@32 : {nmi32_m:.4f} ± {nmi32_s:.4f}")
print(f"MI@16 : {mi16_m:.4f} ± {mi16_s:.4f} | NMI@16 : {nmi16_m:.4f} ± {nmi16_s:.4f}")

latex = r"""
    \begin{table}[t]
    \centering
    \begin{tabular}{c|cc}
    \hline
    Scale & MI ($\uparrow$ dependence) & NMI ($\uparrow$ dependence) \\
    \hline
    64 & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\
    32 & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\
    16 & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\
    \hline
    \end{tabular}
    \caption{Mutual information between identity and attribute heatmaps at different scales. Lower values indicate weaker statistical dependence.}
    \end{table}
    """ % (mi64_m, mi64_s, nmi64_m, nmi64_s,
        mi32_m, mi32_s, nmi32_m, nmi32_s,
        mi16_m, mi16_s, nmi16_m, nmi16_s)

print(latex)


# print(len(os.listdir('/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_64')))
# print(len(os.listdir('/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_32')))
# print(len(os.listdir('/path/to/your/FaceReclaim/train/infoswap-diffface-cs/mask/mask_16')))


    # get_attmap_oppt(id_input64)
    # get_attmap_oppt(id_input32)
    # get_attmap_oppt(id_input16)

    # get_attmap1(att_input64)
    # get_attmap1(att_input32)
    # get_attmap1(att_input16)





