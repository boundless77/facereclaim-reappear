from attribute.model import id_Encoder,att_Encoder,IDExtractor,MappingNetwork,Decoder
import torch
from attribute.data import AttDataset
from torch.optim.lr_scheduler import MultiStepLR
import cv2
import numpy as np
import torch
import os
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

def fix_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # "module."
        new_state_dict[new_key] = value
    return new_state_dict

load_weights = True

dataset = AttDataset(img_path='/path/to/your/attribute_training_images')  # EDIT: training images (00001.jpg, 00002.jpg, ...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
test_dataset = AttDataset(img_path='attribute/data')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device_ids = [0,1]
id_encoder = id_Encoder()
id_encoder = nn.DataParallel(id_encoder, device_ids=device_ids)
if load_weights:
    id_weights = torch.load('attribute/models/id_encoder_epoch_550.pth')
    fixed_id_weights = fix_state_dict_keys(id_weights)
    if len(device_ids) > 1:
        id_encoder.load_state_dict(id_weights)
    else:
        id_encoder.load_state_dict(fixed_id_weights)

att_encoder = att_Encoder()
att_encoder = nn.DataParallel(att_encoder, device_ids=device_ids)
if load_weights:
    att_weights = torch.load('attribute/models/att_encoder_epoch_550.pth')
    fixed_att_weights = fix_state_dict_keys(att_weights)
    if len(device_ids) > 1:
        att_encoder.load_state_dict(att_weights)
    else:
        att_encoder.load_state_dict(fixed_att_weights)

decoder = Decoder()
decoder = nn.DataParallel(decoder, device_ids=device_ids)
if load_weights:
    decoder_weights = torch.load('attribute/models/decoder_epoch_550.pth')
    fixed_decoder_weights = fix_state_dict_keys(decoder_weights)
    if len(device_ids) > 1:
        decoder.load_state_dict(decoder_weights)
    else:
        decoder.load_state_dict(fixed_decoder_weights)


mapping_net = MappingNetwork(input_dim=512 * 8 * 8, output_dim=512)
mapping_net = nn.DataParallel(mapping_net, device_ids=device_ids)
if load_weights:
    mapping_weights = torch.load('attribute/models/mapping_net_epoch_550.pth')
    fixed_mapping_weights = fix_state_dict_keys(mapping_weights)
    if len(device_ids) > 1:
        mapping_net.load_state_dict(mapping_weights)
    else:
        mapping_net.load_state_dict(fixed_mapping_weights)


id_encoder = id_encoder.to(device_ids[0])
att_encoder = att_encoder.to(device_ids[0])
decoder = decoder.to(device_ids[0])
mapping_net = mapping_net.to(device_ids[0])


for batch_idx, (image) in enumerate(dataloader):
    img = image.to(device_ids[0])


    id_features = id_encoder(img)
    att_features = att_encoder(img)
    # id_embedding = mapping_net(id_features[-1])
    # output_image = decoder(id_features, att_features)

    input64 = id_features[1]
    input32 = id_features[2]
    input16 = id_features[3]

    output64 = torch.mean(input64, dim=1)
    output32 = torch.mean(input32, dim=1)
    output16 = torch.mean(input16, dim=1)

    output_tensor_64 = torch.squeeze(output64).cpu()
    output_tensor_32 = torch.squeeze(output32).cpu()
    output_tensor_16 = torch.squeeze(output16).cpu()

    output_numpy_64 = output_tensor_64.detach().numpy()
    output_numpy_32 = output_tensor_32.detach().numpy()
    output_numpy_16 = output_tensor_16.detach().numpy()

    output_numpy = output_numpy_64
    min_val, max_val = np.min(output_numpy), np.max(output_numpy)
    bins = np.linspace(min_val, max_val, 6)
    output_numpy = np.digitize(output_numpy, bins, right=True)
    output_numpy = 1 - (output_numpy - 1) / 4.0
    output_numpy = cv2.resize(output_numpy, (256, 256), interpolation=cv2.INTER_LINEAR)
    plt.imshow(output_numpy, cmap="jet")
    plt.colorbar()
    plt.show()




