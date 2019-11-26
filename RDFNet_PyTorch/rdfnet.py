import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
from blocks import (MMFBlock, RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling, ChainedResidualPool)
from resnet101 import get_resnet101

BatchNorm2d = nn.BatchNorm2d

class RDF(nn.Module):
    def __init__(self, input_size, num_classes, bn_momentum=0.0003, features=256, pretained=False, model_path=''):
        super(RDF, self).__init__()
        self.Resnet101rgb = get_resnet101(bn_momentum=bn_momentum)
        self.Resnet101hha = get_resnet101(bn_momentum=bn_momentum)

        # This is the four stages of each resnet.
        self.rgblayer1 = nn.Sequential(self.Resnet101rgb.conv1, self.Resnet101rgb.bn1, self.Resnet101rgb.relu1,
                                    self.Resnet101rgb.conv2, self.Resnet101rgb.bn2, self.Resnet101rgb.relu2,
                                    self.Resnet101rgb.conv3, self.Resnet101rgb.bn3, self.Resnet101rgb.relu3,
                                    self.Resnet101rgb.maxpool, self.Resnet101rgb.layer1)
        self.rgblayer2 = self.Resnet101rgb.layer2
        self.rgblayer3 = self.Resnet101rgb.layer3
        self.rgblayer4 = self.Resnet101rgb.layer4

        self.hhalayer1 = nn.Sequential(self.Resnet101hha.conv1, self.Resnet101hha.bn1, self.Resnet101hha.relu1,
                                    self.Resnet101hha.conv2, self.Resnet101hha.bn2, self.Resnet101hha.relu2,
                                    self.Resnet101hha.conv3, self.Resnet101hha.bn3, self.Resnet101hha.relu3,
                                    self.Resnet101hha.maxpool, self.Resnet101hha.layer1)
        self.hhalayer2 = self.Resnet101hha.layer2
        self.hhalayer3 = self.Resnet101hha.layer3
        self.hhalayer4 = self.Resnet101hha.layer4

        # MMF Block
        self.mmf1 = MMFBlock(256)
        self.mmf2 = MMFBlock(512)
        self.mmf3 = MMFBlock(1024)
        self.mmf4 = MMFBlock(2048)

        # modify the feature maps from each stage of RenNet, modify their channels
        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)     # here, 2*fetures means we use two same stage-4 features as input

        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, math.ceil(input_size // 32)))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, rgb, hha):
        rgb_layer_1 = self.rgblayer1(rgb)
        rgb_layer_2 = self.rgblayer2(rgb_layer_1)
        rgb_layer_3 = self.rgblayer3(rgb_layer_2)
        rgb_layer_4 = self.rgblayer4(rgb_layer_3)
        
        hha_layer_1 = self.hhalayer1(hha)
        hha_layer_2 = self.hhalayer2(hha_layer_1)
        hha_layer_3 = self.hhalayer3(hha_layer_2)
        hha_layer_4 = self.hhalayer4(hha_layer_3)

        fusion1 = self.mmf1(rgb_layer_1, hha_layer_1)
        fusion2 = self.mmf2(rgb_layer_2, hha_layer_2)
        fusion3 = self.mmf3(rgb_layer_3, hha_layer_3)
        fusion4 = self.mmf4(rgb_layer_4, hha_layer_4)

        # print(fusion1.shape, fusion2.shape, fusion3.shape, fusion4.shape)

        # modify the number of channel
        layer_1_rn = self.layer1_rn(fusion1)
        layer_2_rn = self.layer2_rn(fusion2)
        layer_3_rn = self.layer3_rn(fusion3)
        layer_4_rn = self.layer4_rn(fusion4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(out, size=rgb.size()[-2:], mode='bilinear', align_corners=True)

        return out

def main():
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    left = torch.randn(in_batch, inchannel, in_h, in_w)
    right = torch.randn(in_batch, inchannel, in_h, in_w)
    net = RDF(input_size=in_h, num_classes=num_classes)
    out = net(left, right)
    print(out.shape)

if __name__ == '__main__':
    main()