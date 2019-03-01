'''
[description]
对特征图进行金字塔池化操作。然后将最终特征整合到 256 维
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import resnet101

def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        )
    return asppconv

class AsppModule(nn.Module):
    def __init__(self, out_channels=1280, bn_momentum=0.1, output_stride=16, pooling_size=None):
        super(AsppModule, self).__init__()

        self.pooling_size = pooling_size

        # output_stride choice
        if output_stride ==16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2*[0, 12, 24, 36]
        else:
            raise Warning("output_stride must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        self._atrous_convolution1 = _AsppConv(2048, 256, 1, 1, bn_momentum=bn_momentum)
        self._atrous_convolution2 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution3 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution4 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[3], dilation=atrous_rates[3]
                                              , bn_momentum=bn_momentum)

        self.map_convs = nn.ModuleList([
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.Conv2d(2048, 256, 3, bias=False, dilation=atrous_rates[1], padding=atrous_rates[1]),
            nn.Conv2d(2048, 256, 3, bias=False, dilation=atrous_rates[2], padding=atrous_rates[2]),
            nn.Conv2d(2048, 256, 3, bias=False, dilation=atrous_rates[3], padding=atrous_rates[3])
        ])

        self.red_conv = nn.Conv2d(256 * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(256, out_channels, 1, bias=False)
        self.map_bn = nn.BatchNorm2d(256 * 4)
        self.red_bn = nn.BatchNorm2d(out_channels)


        self.global_pooling_conv = nn.Conv2d(2048, 256, 1, bias=False)
        self.global_pooling_bn = nn.BatchNorm2d(256)

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)        # 每个通道取均值
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (x.shape[2], x.shape[3])         # 验证集上
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = F.avg_pool2d(x, pooling_size, stride=1)
            pool = F.pad(pool, pad=padding, mode="replicate")
        return pool

        self.reset_parameters(self.map_bn.activation, self.map_bn.slope)

    def forward(self, input):
        out = torch.cat([m(input) for m in self.map_convs], dim=1)
        out = self.map_bn(out)      # batch norm
        out = self.red_conv(out)        # modify the number of channels

        # pool = self._image_pool(input)
        pool = self._global_pooling(input)      # global pooling on the raw image
        pool = self.global_pooling_conv(pool)   # change the channel num
        pool = self.global_pooling_bn(pool)

        pool = self.pool_red_conv(pool)

        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, input.size(2), input.size(3))

        out += pool
        out = self.red_bn(out)

        return out

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, InPlaceABNSync):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16, drop_out=True):
        super(Encoder, self).__init__()
        self.ASPP = AsppModule(bn_momentum=bn_momentum, output_stride=output_stride)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout = nn.Dropout(0.5)

        self.__init_weight()
        self.drop_out = drop_out

    def forward(self, input):
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        if self.drop_out:
            input = self.dropout(input)             # here, a drop-out layer is used
        return input


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def main():
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 2048, 28, 28
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = Encoder()
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()