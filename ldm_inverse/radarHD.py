# Core components of U-net

# Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .dice_score import dice_loss
# from .focal_loss import *
__CONSTRAINT_METHOD__ = {}

def register_constraint_method(name: str):
    def wrapper(cls):
        if __CONSTRAINT_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONSTRAINT_METHOD__[name] = cls
        return cls
    return wrapper

def get_constraint_method(name: str, in_channels, out_channels):
    if __CONSTRAINT_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONSTRAINT_METHOD__[name](in_channels, out_channels)


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

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

class Up_nocat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2), mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class Up_nocat_sym(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((8, 8))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=x.shape[-2:], align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat

@register_constraint_method(name='radarHD')
class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.aspp_block = ASPPBlock(in_ch=1024 // factor, out_ch=1024 // factor)

        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(6*(1024 // factor), 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.up5 = Up_nocat(64, 64, bilinear)
        # self.up6 = Up_nocat(64, 64, bilinear)
        # self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        # self.final_sigmoid = nn.Sigmoid()

        self.MSEloss = torch.nn.BCEWithLogitsLoss()
        # self.focal_loss = FocalLoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x_aspp = self.aspp_block(x5)
        x = self.up1(x_aspp, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # x = self.up5(x)
        # x = self.up6(x)
        # x = self.up7(x)
        conv_out = self.outc(x)
        # logits = self.final_sigmoid(conv_out)

        return conv_out
    
    def loss(self, x, gt):
        pred = self.forward(x)
        # MSE loss
        mseloss = self.MSEloss(pred, gt)
        # mseloss = self.focal_loss(pred, gt)
        # DICE loss
        # diceloss = dice_loss(torch.sigmoid(pred), gt)
        # loss = 0.9*mseloss  +0.1*diceloss
        loss = mseloss
        ret = {
            'loss': loss,
            'out': pred,
        }
        return ret
    
