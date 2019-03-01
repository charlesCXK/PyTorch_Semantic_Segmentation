# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# 亮点：在 conv2d 和 bn 之后，增加了 dropout
# 

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet101

from encoder import Encoder

BatchNorm2d = nn.BatchNorm2d

class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1, drop_out=True):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(48, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)         # insert dropout function
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        self._init_weight()
        self.drop_out = drop_out



    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)

        # Enlarge the feature from the Encode Module by 4x.
        x_4 = F.upsample(x, size=low_level_feature.size()[2:4], mode='bilinear' ,align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)

        if self.drop_out:
            x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        if self.drop_out:
            x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, functools.partial):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DeepLabV3plus(nn.Module):
    def __init__(self, output_stride, class_num, pretrained, bn_momentum=0.1, freeze_bn=False, drop_out=True, model_path=''):
        super(DeepLabV3plus, self).__init__()
        self.Resnet101 = resnet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.encoder = Encoder(bn_momentum, output_stride, drop_out)
        self.decoder = Decoder(class_num, bn_momentum, drop_out)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low_level_features = self.Resnet101(input)

        x = self.encoder(x)     # 空间金字塔池化
        predict = self.decoder(x, low_level_features)
        output= F.upsample(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, functools.partial):
                m.eval()


def main():
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = DeepLabV3plus(output_stride=16, class_num=num_classes, pretrained=False)
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()


