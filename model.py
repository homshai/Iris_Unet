# model.py - MobileNetV3 encoder + UNet-style decoder with attention

import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic components

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch, dilation=dilation)
        self.pointwise = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Attention modules

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ASPP module

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2)  # Placeholder, will be dynamically resized
        ))
        
        modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=1))
        for rate in atrous_rates:
            modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=rate))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for i, conv in enumerate(self.convs):
            if i == 0:
                # Global average pooling branch
                out = conv[0](x)  # AdaptiveAvgPool2d(1)
                out = conv[1](out)  # ConvBNReLU
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            else:
                out = conv(x)
            res.append(out)
        
        res = torch.cat(res, dim=1)
        return self.project(res)


# Decoder modules

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch + skip_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch)
        )
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# Main model

class MobileUNetOptimized(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, base_channels=32, use_boundary=False, backbone='mobilenet_v3'):
        super(MobileUNetOptimized, self).__init__()
        self.use_boundary = use_boundary
        self.backbone_type = backbone
        
        # Backbone selection
        if backbone == 'mobilenet_v3':
            from torchvision.models import mobilenet_v3_large
            self.backbone = mobilenet_v3_large(pretrained=pretrained)
            # Extract relevant layers
            self.enc0 = nn.Sequential(self.backbone.features[0], self.backbone.features[1])  # 16x112x112
            self.enc1 = nn.Sequential(*self.backbone.features[2:4])   # 24x56x56
            self.enc2 = nn.Sequential(*self.backbone.features[4:9])   # 40x28x28
            self.enc3 = nn.Sequential(*self.backbone.features[9:14])  # 112x14x14
            self.enc4 = nn.Sequential(*self.backbone.features[14:])   # 960x7x7
            enc_channels = [16, 24, 40, 112, 960]
        elif backbone == 'mobilenet_v2':
            from torchvision.models import mobilenet_v2
            self.backbone = mobilenet_v2(pretrained=pretrained)
            # Extract relevant layers
            self.enc0 = nn.Sequential(self.backbone.features[0], self.backbone.features[1])  # 16x112x112
            self.enc1 = nn.Sequential(*self.backbone.features[2:4])   # 24x56x56
            self.enc2 = nn.Sequential(*self.backbone.features[4:7])   # 32x28x28
            self.enc3 = nn.Sequential(*self.backbone.features[7:14])  # 96x14x14
            self.enc4 = nn.Sequential(*self.backbone.features[14:])   # 1280x7x7
            enc_channels = [16, 24, 32, 96, 1280]
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        
        # Decoder with attention
        base = base_channels
        self.aspp = ASPP(enc_channels[4], base*8)
        self.up4 = UpBlock(base*8, enc_channels[3], base*4)
        self.att4 = CBAM(base*4)
        self.up3 = UpBlock(base*4, enc_channels[2], base*2)
        self.att3 = CBAM(base*2)
        self.up2 = UpBlock(base*2, enc_channels[1], base)
        self.att2 = CBAM(base)
        self.up1 = UpBlock(base, enc_channels[0], base//2)
        self.att1 = CBAM(base//2)
        
        # Final prediction head
        self.final = nn.Conv2d(base//2, num_classes, 1)
        
        # Optional boundary head
        if use_boundary:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(base//2, base//4, 3, padding=1),
                nn.BatchNorm2d(base//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base//4, 1, 1)
            )
    
    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)   # 1/2
        x1 = self.enc1(x0)  # 1/4
        x2 = self.enc2(x1)  # 1/8
        x3 = self.enc3(x2)  # 1/16
        x4 = self.enc4(x3)  # 1/32
        
        # ASPP
        x = self.aspp(x4)
        
        # Decoder with attention
        x = self.up4(x, x3)
        x = self.att4(x)
        x = self.up3(x, x2)
        x = self.att3(x)
        x = self.up2(x, x1)
        x = self.att2(x)
        x = self.up1(x, x0)
        x = self.att1(x)
        
        # Final prediction
        logits = self.final(x)
        
        if self.use_boundary:
            boundary_logits = self.boundary_head(x)
            return logits, boundary_logits
        
        return logits


def build_model(num_classes=1, pretrained=True, base_channels=32, use_boundary=False, backbone='mobilenet_v3'):
    return MobileUNetOptimized(num_classes, pretrained, base_channels, use_boundary, backbone)
