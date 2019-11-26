import math
import sys
import torch
import torch.nn as nn
import torchvision.models as models

from resnet101 import get_resnet101

from blocks import (RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling, ChainedResidualPool)


class BaseRefineNet4Cascade(nn.Module):
    def __init__(self, input_channel, input_size,  refinenet_block, num_classes=1, features=256, resnet_factory=models.resnet101, bn_momentum = 0.01, pretrained=True, freeze_resnet=False):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()

        input_channel = input_channel
        input_size = input_size

        self.Resnet101 = get_resnet101(num_classes=0, bn_momentum=bn_momentum)

        self.layer1 = nn.Sequential(self.Resnet101.conv1, self.Resnet101.bn1, self.Resnet101.relu1,
                                    self.Resnet101.conv2, self.Resnet101.bn2, self.Resnet101.relu2,
                                    self.Resnet101.conv3, self.Resnet101.bn3, self.Resnet101.relu3,
                                    self.Resnet101.maxpool, self.Resnet101.layer1)

        self.layer2 = self.Resnet101.layer2
        self.layer3 = self.Resnet101.layer3
        self.layer4 = self.Resnet101.layer4

        # freeze the resnet parameters, default is false
        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

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

    def forward(self, x):
        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        # modify the number of channel
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return out

    # def named_parameters(self):
    #     """Returns parameters that requires a gradident to update."""
    #     return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 bn_momentum = 0.01,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            bn_momentum = bn_momentum,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


class RefineNet4Cascade(BaseRefineNet4Cascade):
    def __init__(self,
                 input_size,
                 input_channel=3,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 bn_momentum = 0.01,
                 pretrained=True,
                 freeze_resnet=False):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_channel (int): channe num
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_channel,
            input_size,
            RefineNetBlock,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            bn_momentum = bn_momentum,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


def get_refinenet(input_size, num_classes, features=256, bn_momentum=0.01, pretrained=True):
    resnet101 = get_resnet101(num_classes=num_classes, bn_momentum=bn_momentum)      # new ResNet proposed in PSPNet
    return RefineNet4Cascade(input_size, num_classes=num_classes, resnet_factory=resnet101,
                             features=features, bn_momentum=bn_momentum, pretrained=pretrained)

def main():
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = get_refinenet(input_size=in_h, num_classes=num_classes, pretrained=False)
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()