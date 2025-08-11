# model.py - MobileNetV2 encoder + UNet decoder with CBAM and ASPP lightweight modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

# CBAM attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=(kernel_size-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg,mx], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# Simple ASPP-like module
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, kernel=1, padding=0)
        self.conv2 = ConvBNReLU(in_ch, out_ch, kernel=3, padding=1)
        self.conv3 = ConvBNReLU(in_ch, out_ch, kernel=3, padding=1)  # Fixed padding
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = ConvBNReLU(in_ch, out_ch, kernel=1, padding=0)
        self.project = ConvBNReLU(out_ch*4, out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv_pool(self.pool(x))
        x4 = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x1,x2,x3,x4], dim=1)
        out = self.project(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
        self.att = CBAM(out_ch, reduction=8)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.att(x)
        return x

class MobileUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, base_channels=32):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.initial = mobilenet.features[0]
        self.enc1 = nn.Sequential(*mobilenet.features[1:3])
        self.enc2 = nn.Sequential(*mobilenet.features[3:6])
        self.enc3 = nn.Sequential(*mobilenet.features[6:13])
        self.enc4 = nn.Sequential(*mobilenet.features[13:])
        bottleneck_ch = 1280
        self.aspp = ASPP(bottleneck_ch, base_channels*8)
        self.reduce3 = ConvBNReLU(96, base_channels*4)
        self.reduce2 = ConvBNReLU(32, base_channels*2)
        self.reduce1 = ConvBNReLU(24, base_channels)
        self.up3 = UpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.up2 = UpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.up1 = UpBlock(base_channels*2, base_channels, base_channels)
        self.final_conv = nn.Sequential(
            ConvBNReLU(base_channels, base_channels),
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        r4 = self.aspp(x4)
        r3 = self.reduce3(x3)
        r2 = self.reduce2(x2)
        r1 = self.reduce1(x1)
        d3 = self.up3(r4, r3)
        d2 = self.up2(d3, r2)
        d1 = self.up1(d2, r1)
        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out  # logits
def build_model(num_classes=1, pretrained=True, base_channels=32):
    return MobileUNet(num_classes=num_classes, pretrained=pretrained, base_channels=base_channels)
