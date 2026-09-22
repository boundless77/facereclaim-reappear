import torch
import torch.nn as nn
import torch.nn.functional as F


def _initialize_weights(self):

    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        self._initialize_weights()

class IDExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.netArc_checkpoint = 'checkpoints/arcface_checkpoint.tar'  # put the ArcFace checkpoint here (see readme)
        self.netArc_checkpoint = torch.load(self.netArc_checkpoint)
        self.netArc = self.netArc_checkpoint
        self.device = device
        self.netArc.eval()
        self.netArc.requires_grad_(False)

    def forward(self, x):
        model = self.netArc
        model.eval()
        images_id_downsample = F.interpolate(x, size=(112, 112), mode='bicubic')
        images_latend_id = model(images_id_downsample)
        id = F.normalize(images_latend_id, p=2, dim=1)

        return id



  # Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()
  # 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.inst_norm = nn.InstanceNorm2d(out_channels)
  # LeakyReLU
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
  # LeakyReLU
        x = self.conv(x)
        x = self.inst_norm(x)
        x = self.leaky_relu(x)
        return x



  # Encoder
class id_Encoder(nn.Module):
    def __init__(self, input_channels=3, feature_channels=512):
        super(id_Encoder, self).__init__()
  # ConvBlock
        self.blocks = nn.ModuleList([
            ConvBlock(input_channels, 64),  # (3, 256, 256) -> (64, 128, 128)
            ConvBlock(64, 128),  # (64, 128, 128) -> (128, 64, 64)
            ConvBlock(128, 256),  # (128, 64, 64) -> (256, 32, 32)
            ConvBlock(256, 512),  # (256, 32, 32) -> (512, 16, 16)
            ConvBlock(512, feature_channels)  # (512, 16, 16) -> (C, 8, 8)
        ])
    def forward(self, x):
  # 5block
        id_features = []
        for block in self.blocks:
            x = block(x)
            id_features.append(x)
        return id_features

class att_Encoder(nn.Module):
    def __init__(self, input_channels=3, feature_channels=512):
        super(att_Encoder, self).__init__()

  # ConvBlock
        self.blocks = nn.ModuleList([
            ConvBlock(input_channels, 64),  # (3, 256, 256) -> (64, 128, 128)
            ConvBlock(64, 128),  # (64, 128, 128) -> (128, 64, 64)
            ConvBlock(128, 256),  # (128, 64, 64) -> (256, 32, 32)
            ConvBlock(256, 512),  # (256, 32, 32) -> (512, 16, 16)
            ConvBlock(512, feature_channels)  # (512, 16, 16) -> (C, 8, 8)
        ])
    def forward(self, x):
  # 5block
        att_features = []
        for block in self.blocks:
            x = block(x)
            att_features.append(x)
        return att_features

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.inst_norm = nn.InstanceNorm2d(out_channels)
  # LeakyReLU
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, scale_factor=2):

        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
  # LeakyReLU
        x = self.conv(x)
        x = self.inst_norm(x)
        x = self.leaky_relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, feature_channels=512, output_channels=3):
        super(Decoder, self).__init__()

  # UpConvBlock
        self.blocks = nn.ModuleList([
            UpConvBlock(feature_channels * 2, 1024),
            UpConvBlock(512 * 2, 512),
            UpConvBlock(256 * 2, 256),
            UpConvBlock(128 * 2, 128),
            UpConvBlock(64 * 2, 64),
        ])

  # 3
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()  # [-1, 1]

    def forward(self, id_features, att_features):
  # features idatt
        features = [torch.cat((id_f, att_f), dim=1) for id_f, att_f in zip(id_features, att_features)]

  # 8x8
        x = features[-1]  # 8×8


        for i in range(4, 0, -1):
            x = self.blocks[4 - i](x)
            x = x + features[i - 1]

        x = self.blocks[-1](x)

        x = self.final_conv(x)
        x = self.tanh(x)  # Tanh[-1, 1]
        return x

class MappingNetwork(nn.Module):
    def __init__(self, input_dim=512 * 8 * 8, output_dim=512):
        super(MappingNetwork, self).__init__()
  # B×1×512
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
  # batch
        x = x.view(x.size(0), -1)  # B×(512*8*8)
        x = self.fc(x)  # B×512
        return x



  # Encoder
if __name__ == '__main__':

    id_encoder = id_Encoder()
    att_encoder = att_Encoder()
    decoder = Decoder()
    mapping_net = MappingNetwork(input_dim=512 * 8 * 8, output_dim=512)
    IDExtractor = IDExtractor()
    input_image = torch.randn(4, 3, 256, 256)

    id_true = IDExtractor(input_image)
    id_features = id_encoder(input_image)
    att_features = att_encoder(input_image)


    output_image = decoder(id_features, att_features)
    print(output_image.shape)  # check output shape (1, 3, 256, 256)


    embedding = mapping_net(id_features[-1])


    print("Mapped output shape:", embedding.shape)  # expected (1, 512)




