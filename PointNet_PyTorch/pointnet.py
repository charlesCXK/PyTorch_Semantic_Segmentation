# -*- coding: utf-8 -*-
# @Time    : 2019/01/24
# @Author  : Chen Xiaokang
# @Email   : pkucxk@pku.edu.cn
# @File    : pointnet.py
# @Software: PyCharm

import os
import sys
import numpy as np
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from feat_transform import *

'''
[description]
PointNetClassify
PointNetSegmentation
input is Bx3xN
        Return:
            BxNxK (K is the class num)
'''
class PointNetClassify(nn.Module):
    def __init__(self, class_num=2):
        super(PointNetClassify, self).__init__()
        self.class_num = class_num
        self.input_transform = input_transform_net()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.feature_transform = feature_transform_net(K=64)

        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.7)

        self.fc3 = nn.Linear(256, self.class_num)

        self.classifier = torch.nn.Conv1d(128, self.class_num, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]  # number of points per sample

        input_transform = self.input_transform(x)
        x = x.transpose(2, 1)  # Bx3xN -> BxNx3
        x = torch.bmm(x, input_transform)  # bmm: batch matrix multiply; x -> BxNx3
        x = x.transpose(2, 1)  # BxNx3 -> Bx3xN

        # mlp(64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))     # -> Bx64xN

        feature_transform = self.feature_transform(x)
        x = x.transpose(2, 1)  # Bx64xN -> BxNx64
        x = torch.bmm(x, feature_transform)  # bmm: batch matrix multiply; x -> BxNx64
        x = x.transpose(2, 1)  # BxNx64 -> Bx64xN

        # mlp(64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))     # -> Bx1024xN

        x = torch.max(x, 2, keepdim=True)[0]
        global_feat = x.view(-1, 1024)  # global features, Bx1024

        x = F.relu(self.fc_bn1(self.fc1(global_feat)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)
        return x

'''
[description]
PointNetSegmentation
input is Bx3xN
        Return:
            BxNxK (K is the class num)
'''
class PointNetSegmentation(nn.Module):
    def __init__(self, class_num=2, global_feat=False):
        super(PointNetSegmentation, self).__init__()
        self.class_num = class_num
        self.global_feat = global_feat

        self.input_transform = input_transform_net()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.feature_transform = feature_transform_net(K=64)

        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        # segmentation network
        if not self.global_feat:
            self.conv6 = torch.nn.Conv1d(1088, 512, 1)
            self.bn6 = nn.BatchNorm1d(512)
        else:
            self.conv6 = torch.nn.Conv1d(1024, 512, 1)
            self.bn6 = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv8 = torch.nn.Conv1d(256, 128, 1)
        self.bn8 = nn.BatchNorm1d(128)

        self.classifier = torch.nn.Conv1d(128, self.class_num, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]  # number of points per sample

        input_transform = self.input_transform(x)
        x = x.transpose(2, 1)  # Bx3xN -> BxNx3
        x = torch.bmm(x, input_transform)  # bmm: batch matrix multiply; x -> BxNx3
        x = x.transpose(2, 1)  # BxNx3 -> Bx3xN

        # mlp(64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))     # -> Bx64xN

        feature_transform = self.feature_transform(x)
        x = x.transpose(2, 1)  # Bx64xN -> BxNx64
        x = torch.bmm(x, feature_transform)  # bmm: batch matrix multiply; x -> BxNx64
        x = x.transpose(2, 1)  # BxNx64 -> Bx64xN

        # to save memory
        del feature_transform
        del input_transform

        # Feature for per point, local feature
        pointfeat = x

        # mlp(64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))     # -> Bx1024xN

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # global features, Bx1024

        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)  # Bx1024 -> Bx1024xN

        # Global-feature + point-feature
        if not self.global_feat:
            x =  torch.cat([x, pointfeat], 1)

        # mlp(512, 256, 128)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        # mlp(128, m)
        x = self.classifier(x)

        x = x.transpose(2, 1).contiguous()

        x = F.log_softmax(x.view(-1, self.class_num), dim=-1)
        x = x.view(batchsize, n_pts, self.class_num)
        return x

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))

    cly = PointNetClassify(class_num = 40)
    out = cly(sim_data)
    print('cly', out.size())

    seg = PointNetSegmentation(class_num = 40)
    out = seg(sim_data)
    print('seg', out.size())