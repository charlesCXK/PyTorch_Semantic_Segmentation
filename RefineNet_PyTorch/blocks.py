import torch.nn as nn

'''
[description]
RCU block
'''
class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

'''
[description]

'''
class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])       # get the maxer shape of several input feture maps
        self.max_size = (max_size, max_size)

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            # if max_size % size != 0:
            #     raise ValueError("max_size not divisble by shape {}".format(i))

            # self.scale_factors.append(max_size // size)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):
        # print(self.max_size)
        max_size = self.max_size#xs[-1].size()[-2:]     # max size of these feature, in default situation, the last data in the data-array has the biggest shape
        output = self.resolve0(xs[0])
        if xs[0].size()[-2] != max_size[0]:
            output = nn.functional.interpolate(
                output,
                size=max_size,
                mode='bilinear',
                align_corners=True)

        for i, x in enumerate(xs[1:], 1):
            this_feature = self.__getattr__("resolve{}".format(i))(x)
            # upsamples all (smaller) feature maps to the largest resolution of the inputs
            if xs[i].size()[-2] != max_size[0]:
                this_feature = nn.functional.interpolate(
                    this_feature,
                    size=max_size,
                    mode='bilinear',
                    align_corners=True)
            output += this_feature

        return output


'''
[description]
chained residual pool
'''
class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        # two pool-block
        for i in range(1, 3):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),   # obtain the raw feature map size
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 3):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]        # channel-num of this stage's output feature map
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        # stage-4 of ResNet needn't have to use 'multi_resolution_fusion'
        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        # multi-resolution input fusion
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        # Multi-resolution Fusion
        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        # Chained Residual Pooling
        out = self.crp(out)

        # Output Conv.
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

class MMFBlock(nn.Module):
    def __init__(self, features):
        super(MMFBlock, self).__init__()
        self.downchannel = features // 2

        self.relu = nn.ReLU(inplace=True)

        self.rgb_feature = nn.Sequential(
            nn.Conv2d(features, self.downchannel, kernel_size=1, stride=1, padding=0, bias=False),      # downsample

            # nonlinear_transformations
            ResidualConvUnit(self.downchannel),
            ResidualConvUnit(self.downchannel),

            nn.Conv2d(self.downchannel, features, kernel_size=3, stride=1, padding=1, bias=False)       # upsample
        )
        self.hha_feature = nn.Sequential(
            nn.Conv2d(features, self.downchannel, kernel_size=1, stride=1, padding=0, bias=False),      # downsample

            # nonlinear_transformations
            ResidualConvUnit(self.downchannel),
            ResidualConvUnit(self.downchannel),

            nn.Conv2d(self.downchannel, features, kernel_size=3, stride=1, padding=1, bias=False)       # upsample
        )

        self.ResidualPool = nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),   # obtain the raw feature map size
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))

    def forward(self, rgb, hha):
        rgb_fea = self.rgb_feature(rgb)
        hha_fea = self.hha_feature(hha)
        fusion = self.relu(rgb_fea + hha_fea)
        x = self.ResidualPool(fusion)
        return fusion + x



