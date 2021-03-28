
from torch import nn, Tensor
from typing import Dict
import torch
from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


__all__ = ["LRASPP"]


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(self, stem, backbone, low_channels, high_channels, num_classes, inter_channels=128):
        super().__init__()
        self.stem = stem
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input):
        input_ = self.stem(input)
        features = self.backbone(input_)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode='bilinear', align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):

    def __init__(self, low_channels, high_channels, num_classes, inter_channels):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def get_layer_by_unique_name(module, unique_name):
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        return module._modules[unique_names[0]]
    else:
        return get_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
        )


def convert2segmentation(model: nn.Module, begin_index_index, num_classes=19):
    for name, module in model.named_modules():
        try:
            index = int(name.split(".")[1])
        except Exception:
            continue
        if index >= begin_index_index:
            if "depth_conv.conv" in name:
                if isinstance(module, nn.Conv2d):
                    module.stride = (1, 1)
                    module.dilation = (2, 2)
                    padding = (module.kernel_size[0] - 1) // 2 * module.dilation[0]
                    module.padding = (padding, padding)
    low_pos = 8
    high_pos = 20

    low_channels = get_layer_by_unique_name(model.blocks, f"{low_pos}.mobile_inverted_conv.point_linear.conv").out_channels
    high_channels = get_layer_by_unique_name(model.blocks, f"{high_pos}.mobile_inverted_conv.point_linear.conv").out_channels
    backbone = IntermediateLayerGetter(model.blocks,
                                       return_layers={str(low_pos): 'low', str(high_pos): 'high'})

    segmentation_model = LRASPP(model.first_conv, backbone, low_channels, high_channels, num_classes)
    return segmentation_model
