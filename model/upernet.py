"""
@ github: https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
@ paper: https://arxiv.org/pdf/1807.10221.pdf
@ model: UperNet
@ author: Baoying Chen
@ time: 2022/2/24
"""
import os

os.environ['TORCH_HOME'] = '/data2/hulh/pretrain'
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .backbone.convnext import get_convnext
import numpy as np


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, out_channel=None, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channel if out_channel else in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel if out_channel else in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.4),
                nn.ReLU(inplace=True)
            )
        )
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[-2:], mode='bilinear', align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Softmax(dim=1)):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)


class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=1, in_channels=3, backbone='resnet101', pretrained=True,
                 activation=nn.Sigmoid(), use_edge=False, scale=4, input_size=224, efn_start_down=True, use_roi=False,
                 use_double_conv=False, freeze_backbone=False, is_fpn=True):
        super(UperNet, self).__init__()
        self.use_edge = use_edge
        self.use_roi = use_roi
        self.convnext_scale = scale  # UperNet第一个卷积缩放的比例
        self.use_double_conv = use_double_conv
        self.freeze_backbone = freeze_backbone  # 是否冻结主干网络
        self.is_fpn = is_fpn

        if 'convnext' in backbone:
            self.backbone = get_convnext(in_chans=in_channels, model_name=backbone, pretrained=pretrained,
                                         scale=self.convnext_scale)
            feature_channels = self.backbone.dims

        fpn_out = feature_channels[0]
        self.PPN = PSPModule(feature_channels[-1])
        if self.is_fpn:
            self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
            self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        else:
            self.up1 = Up(feature_channels[-1], feature_channels[-2])
            self.up2 = Up(feature_channels[-2], feature_channels[-3])
            self.up3 = Up(feature_channels[-3], feature_channels[-4])
            self.head = DoubleConv(np.array(self.backbone.dims).sum(), num_classes)


        if self.use_double_conv:
            self.double_conv = DoubleConv(num_classes, num_classes)

        self.outc = OutConv(num_classes, num_classes, activation)
        if self.use_edge:
            self.outc_edge = OutConv(num_classes, num_classes, activation)
        if self.use_roi:
            self.outc_roi = OutConv(num_classes, num_classes, activation)

        self._freeze_backbone()  # 冻结主干网络

    def forward(self, x, label=None, **kwargs):
        B, _, H, W = x.shape
        # input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)


        # print([feature.shape for feature in features])
        features[-1] = self.PPN(features[-1])
        if self.is_fpn:
            x = self.FPN(features)
        else:
            P = [features[-1]]
            x1 = self.up1(features[-1], features[-2])
            P.append(x1)
            x2 = self.up2(x1, features[-3])
            P.append(x2)
            x3 = self.up3(x2, features[-4])
            P.append(x3)
            P = [F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) for x in P]
            x = torch.cat(P, dim=1)
        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)


        if self.use_double_conv:
            x = self.double_conv(x)
        logits = self.outc(x)

        out = [logits]
        if self.use_edge:
            edge_logits = self.outc_edge(x)
            out.append(edge_logits)
        if self.use_roi:
            roi_logits = self.outc_roi(x)
            out.append(roi_logits)
            logits = logits * torch.sigmoid(roi_logits)
            out[0] = logits

        return out if len(out) > 1 else out[0]

    def _freeze_backbone(self):
        if self.freeze_backbone:
            self.backbone.eval()
            for params in self.backbone.parameters():
                params.requires_grad = False

