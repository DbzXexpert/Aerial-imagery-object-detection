import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super(Conv2dReLU, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=not use_batchnorm),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture with ResNet34 encoder.
    """

    def __init__(self, num_classes=1, pretrained=True, use_batchnorm=True, freeze_encoder=False):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        resnet = models.resnet34(pretrained=pretrained)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            self.pool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        encoder_channels = [512, 256, 128, 64, 64]
        decoder_channels = [256, 128, 64, 32, 16]
        in_channels = self.compute_channels(encoder_channels, decoder_channels)

        self.layers = nn.ModuleList([
            DecoderBlock(in_channels[i], decoder_channels[i], use_batchnorm=use_batchnorm)
            for i in range(len(encoder_channels))
        ])

        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[i] + decoder_channels[i]
            for i in range(len(encoder_channels))
        ]
        return channels

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        features = features[::-1]

        for i, layer in enumerate(self.layers):
            features[i] = layer([features[i], features[i + 1] if i + 1 < len(features) else None])

        x = self.final_conv(features[0])
        return torch.sigmoid(x)
