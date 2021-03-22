# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict
import copy
import torch.nn as nn
from torch.nn import functional as F
from .ofa.utils.layers import set_layer_from_config, MBConvLayer, ConvLayer, IdentityLayer, LinearLayer, ResidualBlock
from .ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d

__all__ = ['MobileNetV3', 'MobileNetV3Segmentation']


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)
        return x

    def set_bn_param(self, bn_momentum=0.1, bn_eps=0.00001):
        for m in self.modules():
            if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
                m.momentum = bn_momentum
                m.eps = bn_eps
        return


class MobileNetV3Segmentation(MyNetwork):

    def __init__(self, first_conv, blocks, remain_block, head, aux_heaad):
        super(MobileNetV3Segmentation, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.Sequential(*blocks)
        self.remain_block = nn.Sequential(*remain_block)
        self.head = head
        self.aux_head = aux_heaad
        # self.final_expand_layer = final_expand_layer
        # self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
        # self.feature_mix_layer = feature_mix_layer
        # self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        x = self.first_conv(x)
        intermidiate_features = self.blocks(x)
        # print(intermidiate_features.shape)
        features = self.remain_block(intermidiate_features)

        result = OrderedDict()
        x = features
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_head is not None:
            x = intermidiate_features
            x = self.aux_head(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

    def set_bn_param(self, bn_momentum=0.1, bn_eps=0.00001):
        for m in self.modules():
            if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
                m.momentum = bn_momentum
                m.eps = bn_eps
        return
