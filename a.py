from convert_seg import convert2segmentation
from ofa.ofa_mbv3 import OFAMobileNetV3
from spos_ofa_segmentation.representation import OFAArchitecture
import torch
import torch.nn as nn
# supernet = OFAMobileNetV3(width_mult_list=1.2,)

from torchvision.models.segmentation import lraspp_mobilenet_v3_large

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
            print(name, f"c={m.out_channels}", f"size={h}*{w}", tmp/1e6)
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

supernet = OFAMobileNetV3(
    n_classes=1000,
    dropout_rate=0,
    width_mult_list=1.2,
    ks_list=[3, 5, 7],
    expand_ratio_list=[3, 4, 6],
    depth_list=[2, 3, 4],
)
a = OFAArchitecture.from_legency_string("4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6")

supernet.set_active_subnet(ks=a.ks, e=a.ratios, d=a.depths)
# print(a.ks, a.ratios, a.depths)
# print(supernet.sample_active_subnet())
model = supernet.get_active_subnet()

s = torch.load("model_best.pth.tar", map_location="cpu")
print(s.keys())

model.load_state_dict(s["state_dict_ema"])

for name, module in model.blocks.named_children():
    print(name)

# x = torch.rand(1, 3, 224, 224)
# y = model(x)
# print(y.shape)
# print(model)


seg_model = convert2segmentation(model=model, begin_index_index=17)
# print(seg_model)
import torch
x = torch.rand(1, 3, 224, 224)
y = seg_model(x)
# compute_flops(seg_model,(1, 3, 224, 224))
compute_flops(lraspp_mobilenet_v3_large(),(1, 3, 224, 224))
print(y["out"].shape)
# print(model)