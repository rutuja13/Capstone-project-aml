from efficientnet_pytorch import EfficientNet

import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_efficientnet(n_in, n_out):
    return EfficientNet.from_name("efficientnet-b3", in_channels=n_in, num_classes=n_out)


def get_resnet50(n_in, n_out):
    model = models.resnet50()
    model.conv1 = nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=n_out, bias=True)
    return model


def get_densenet121(n_in, n_out):
    model = models.densenet121()
    model.features.conv0 = nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(in_features=1024, out_features=n_out, bias=True)
    return model


def get_inceptionv3(n_in, n_out):
    model = models.inception_v3(aux_logits=False, init_weights=False)
    model.Conv2d_1a_3x3.conv = nn.Conv2d(n_in, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=n_out, bias=True)
    return model


def get_mobilenetv2(n_in, n_out):
    model = models.mobilenet_v2()
    model.features[0][0] = nn.Conv2d(n_in, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=n_out, bias=True)
    return model


def get_resnext50_32x4d(n_in, n_out):
    model = models.resnext50_32x4d()
    model.conv1 = nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=n_out, bias=True)
    return model


def get_wide_resnet_50_2(n_in, n_out):
    model = models.wide_resnet50_2()
    model.conv1 = nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=n_out, bias=True)
    return model


CLASSIFIER_MODEL_GENERATORS = {
    'efficientnet': get_efficientnet,
    'resnet50': get_resnet50,
    'inceptionv3': get_inceptionv3,
    'densenet121': get_densenet121,
    'mobilenetv3': get_mobilenetv2,
    'resnext50': get_resnext50_32x4d,
    'wideresnet50': get_wide_resnet_50_2
}


# UNet model from https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_unet(n_in, n_out):
    return UNet(n_channels=n_in, n_classes=n_out)
