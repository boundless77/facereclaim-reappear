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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
test_dataset = AttDataset(img_path='attribute/data')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device_ids = [0,1,2]
id_encoder = id_Encoder()
id_encoder = nn.DataParallel(id_encoder, device_ids=device_ids)
if load_weights:
    id_weights = torch.load('attribute/models/id_encoder_epoch_550.pth')
    id_encoder.load_state_dict(id_weights)

att_encoder = att_Encoder()
att_encoder = nn.DataParallel(att_encoder, device_ids=device_ids)
if load_weights:
    att_weights = torch.load('attribute/models/att_encoder_epoch_550.pth')
    att_encoder.load_state_dict(att_weights)

decoder = Decoder()
decoder = nn.DataParallel(decoder, device_ids=device_ids)
if load_weights:
    decoder_weights = torch.load('attribute/models/decoder_epoch_550.pth')
    decoder.load_state_dict(decoder_weights)

mapping_net = MappingNetwork(input_dim=512 * 8 * 8, output_dim=512)
mapping_net = nn.DataParallel(mapping_net, device_ids=device_ids)
if load_weights:
    mapping_weights = torch.load('attribute/models/mapping_net_epoch_550.pth')
    mapping_net.load_state_dict(mapping_weights)


netArc = IDExtractor()
netArc = nn.DataParallel(netArc, device_ids=device_ids)
netArc.eval()
netArc.requires_grad_(False)


params_train = (
    list(id_encoder.parameters())
    + list(att_encoder.parameters())
    + list(decoder.parameters())
    + list(mapping_net.parameters())
)
optimizer = optim.Adam(params_train, lr=1e-6, betas=(0.9, 0.999), eps=1e-8)
scheduler = MultiStepLR(optimizer, milestones=[500, 200], gamma=0.1)


writer = SummaryWriter(log_dir='attribute/logs')


id_encoder = id_encoder.to(device_ids[0])
att_encoder = att_encoder.to(device_ids[0])
decoder = decoder.to(device_ids[0])
mapping_net = mapping_net.to(device_ids[0])
netArc = netArc.to(device_ids[0])


recLoss = nn.L1Loss()


def cosin_metric(x1, x2):
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


import torch
import torch.nn as nn


def negative_correlation_loss(att_features, id_features):
    loss = 0
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
        loss += (1-correlation).mean()
    return loss


for epoch in range(0, 1000):
    rec_loss_total = 0.0
    id_loss_total = 0.0
    att_loss_total = 0.0
    num_batches = len(dataloader)


    for batch_idx, (image) in enumerate(dataloader):
        img = image.to(device_ids[0])


        id_features = id_encoder(img)
        att_features = att_encoder(img)
        id_true = netArc(img)
        id_embedding = mapping_net(id_features[-1])
        output_image = decoder(id_features, att_features)


        rec_loss = recLoss(img, output_image)
        id_loss = (1 - cosin_metric(id_true, id_embedding)).mean()
        att_loss = negative_correlation_loss(att_features, id_features)

        total_loss = rec_loss + id_loss + att_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        rec_loss_total += rec_loss.item()
        id_loss_total += id_loss.item()
        att_loss_total += att_loss.item()

        if batch_idx % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{1000}], Step [{batch_idx+1}/{len(dataloader)}], '
                f'Rec Loss: {rec_loss.item():.4f}, ID Loss: {id_loss.item():.4f}, '
                f'Att Loss: {att_loss.item():.4f}, Total Loss: {total_loss.item():.4f}'
            )


    scheduler.step()

  # TensorBoard
    writer.add_scalar('Loss/rec_loss', rec_loss_total / len(dataloader), epoch)
    writer.add_scalar('Loss/id_loss', id_loss_total / len(dataloader), epoch)
    writer.add_scalar('Loss/att_loss', att_loss_total / len(dataloader), epoch)
    writer.add_scalar('Loss/total_loss', (rec_loss_total + id_loss_total + att_loss_total) / len(dataloader), epoch)

    with torch.no_grad():
        for batch_idx, (image) in enumerate(test_dataloader):
            eval_img = image.to(device_ids[0])
            eval_id_features = id_encoder(eval_img)
            eval_att_features = att_encoder(eval_img)
            eval_output = decoder(eval_id_features, eval_att_features)
            eval_output = torch.clamp((eval_output + 1.0) / 2.0, min=0.0, max=1.0)
            eval_output = 255.0 * rearrange(eval_output, "1 c h w -> h w c")
            eval_output = Image.fromarray(eval_output.type(torch.uint8).cpu().numpy())
            eval_output.save(f'attribute/eval/epoch_{epoch+1}_sample_{batch_idx+1}.jpg')


    if (epoch + 1) % 50 == 0:
        torch.save(id_encoder.state_dict(), f'attribute/models/id_encoder_epoch_{epoch+1}.pth')
        torch.save(att_encoder.state_dict(), f'attribute/models/att_encoder_epoch_{epoch+1}.pth')
        torch.save(decoder.state_dict(), f'attribute/models/decoder_epoch_{epoch+1}.pth')
        torch.save(mapping_net.state_dict(), f'attribute/models/mapping_net_epoch_{epoch+1}.pth')

print('Training complete')
























