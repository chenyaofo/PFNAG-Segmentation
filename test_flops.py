import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import presets
import utils
from spos_ofa_segmentation import SPOSMobileNetV3Segmentation
from spos_ofa_segmentation.representation import OFAArchitecture
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def compute_nparam(module: nn.Module) -> int:
    """Count how many parameter in a module. Note that the buffer in the module will not
    be counted.
    Args:
        module (nn.Module): The module to be counted.
    Returns:
        int: The number of parameters in the module.
    """
    return sum(map(lambda p: p.numel(), module.parameters()))

def compute_flops(module: nn.Module, size: int) -> int:
    """Compute the #MAdds of a module. The current version of this function can only compute the
    #MAdds of nn.Conv2d and nn.Linear. Besides, the input of the module should be a single tensor.
    Args:
        module (nn.Module): The module to be computed.
        size (int): The size of the input tensor.
    Returns:
        int: The number of MAdds.
    """
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print(name)
            hooks.append(m.register_forward_hook(size_hook))
    # print()
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size))
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if "aux_head" in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            tmp = h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            print(name, f"{h}*{w}", tmp/1e6)
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


supernet = SPOSMobileNetV3Segmentation(width_mult=1.2)

archs = [
    # "2,3,3,3,3:3,3,0,0,3,5,3,0,3,3,5,0,3,3,5,0,5,5,7,0:3,3,0,0,4,3,3,0,4,4,4,0,4,6,6,0,6,6,3,0",
    # "2,3,3,4,4:5,3,0,0,5,3,3,0,3,5,5,0,5,5,5,3,7,5,7,5:3,4,0,0,3,4,4,0,3,3,6,0,4,6,6,6,6,6,6,6",
    # "2,4,4,4,4:5,3,0,0,5,5,5,3,5,5,5,5,3,5,5,5,7,5,5,5:4,3,0,0,4,4,4,6,6,3,6,3,6,6,4,6,6,6,6,3",
    # "3,4,4,4,4:5,3,7,0,5,5,5,5,5,5,5,5,3,5,5,3,7,5,5,5:4,4,6,0,4,4,6,6,6,4,6,6,6,6,4,6,6,6,6,6",
    "4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6",
]
for arch in archs:
    model = supernet.get_subnet(OFAArchitecture.from_legency_string(arch))
    # print(model)
    # resolutions = [(1, 3, 1024, 512), (1, 3, 769, 769), (1, 3, 2048, 1024)]
    resolutions = [(1, 3, 1024, 512)]

    for r in resolutions:
        with torch.no_grad():
            param = compute_nparam(model)
            rev = compute_flops(model, r)
            print(arch, r, rev/1e6,param/1e6)
    print("---")
# r =(1, 3, 1024, 512)
r =(1, 3, 2048, 1024)
model = lraspp_mobilenet_v3_large(num_classes=19)

# print(model)
param = compute_nparam(model)
rev = compute_flops(model, r)
print(  r, rev/1e6,param/1e6)