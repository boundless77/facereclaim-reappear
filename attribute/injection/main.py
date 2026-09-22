from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
from models import *
from config import Config
import torch
import os
import torch.nn.parallel
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
import random
from models.fs_networks import Generator_Adain_Upsample
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

def fix_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # "module."
        new_state_dict[new_key] = value
    return new_state_dict
transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def cosin_metric(x1, x2):
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))
def fix_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # "module."
        new_state_dict[new_key] = value
    return new_state_dict
img_folder = '/path/to/your/attribute'

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
        loss += (1 + correlation).mean()
    return loss

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

        # Load color image using cv2
        color_image = cv2.imread(img_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = rearrange(2 * torch.tensor(color_image).float() / 255 - 1, "h w c -> c h w")

        # Load grayscale image using cv2
        grayscale_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        grayscale_image = cv2.resize(grayscale_image, (128, 128))
        grayscale_image = 2 * torch.tensor(grayscale_image).float() / 255 - 1
        grayscale_image = grayscale_image.unsqueeze(0)  # Add channel dimension
        return color_image, grayscale_image


dataset = AttDataset(img_path='/path/to/your/attribute/')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=80, shuffle=True)

test_dataset = AttDataset(img_path='/path/to/your/attribute/data/')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device_ids = [0, 1, 2]
opt = Config()
model = resnet_face18(opt.use_se)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load(opt.test_model_path))
model.eval()
model.requires_grad_(False)
load_weights = True
net_G = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9)
net_G = nn.DataParallel(net_G, device_ids=device_ids)
if load_weights:
    weights = torch.load("/path/to/your/attribute/injection/weights/G_net_epoch_100.pth")
    net_G.load_state_dict(weights)


recLoss = nn.L1Loss()


params_train = (list(net_G.parameters()))

optimizer = optim.Adam(params_train, lr=1e-6, betas=(0.9, 0.999), eps=1e-8)
scheduler = MultiStepLR(optimizer, milestones=[500, 700], gamma=0.1)


writer = SummaryWriter(log_dir='/path/to/your/attribute/injection/logs/')

model = model.to(device_ids[0])
net_G = net_G.to(device_ids[0])

for epoch in range(101, 1000):
    rec_loss_total = 0
    id_loss_total = 0
    att_loss_total = 0

    for batch_idx, (color_image, grayscale_image) in enumerate(dataloader):
        color_image = color_image.to(device_ids[0])
        grayscale_image = grayscale_image.to(device_ids[0])

        id, id_features = model(grayscale_image)

        color_output, att_features = net_G(color_image, id)
        gray_output = TF.rgb_to_grayscale(color_output)
        resized_gray_output = F.interpolate(gray_output, size=(128, 128), mode='bilinear', align_corners=False)

        att_loss = negative_correlation_loss(att_features, id_features)
        rec_loss = recLoss(color_output, color_image)

        rec_id, _ = model(resized_gray_output)
        id_loss = (1 - cosin_metric(id, rec_id)).mean()

        total_loss = rec_loss + id_loss + att_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        rec_loss_total += rec_loss.item()
        id_loss_total += id_loss.item()
        att_loss_total += att_loss.item()

        if batch_idx % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{1000}], Step [{batch_idx + 1}/{len(dataloader)}], '
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
        for batch_idx, (color_image, grayscale_image) in enumerate(test_dataloader):
            color_image = color_image.to(device_ids[0])
            grayscale_image = grayscale_image.to(device_ids[0])
            id, _ = model(grayscale_image)
            output_image, _ = net_G(color_image, id)
            output_image = torch.clamp((output_image + 1.0) / 2.0, min=0.0, max=1.0)
            output_image = 255.0 * rearrange(output_image, "1 c h w -> h w c")
            output_image = Image.fromarray(output_image.type(torch.uint8).cpu().numpy())
            output_image.save(f'/path/to/your/attribute/injection/eval/epoch_{epoch + 1}_sample_{batch_idx + 1}.jpg')



    if (epoch + 1) % 50 == 0:
        torch.save(net_G.state_dict(), f'/path/to/your/attribute/injection/weights/G_net_epoch_{epoch + 1}.pth')

print('Training complete')





