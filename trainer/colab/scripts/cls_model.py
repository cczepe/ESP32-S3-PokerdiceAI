"""Small classification network (from scratch), only building blocks ESP-DL supports:
Conv, (depthwise conv), BatchNorm (folded in when converting), ReLU, MaxPool, GlobalAveragePool, Linear.

Structure: 4 stages [2× (Conv3×3-BN-ReLU)] + MaxPool, then GlobalAveragePool and Linear.
Channels = base [32, 64, 128, 256] × width, rounded up to multiples of 16 (SIMD processes 16 int8 at once).
  width 0.25 -> [16, 16, 32, 64]  ≈ 80 000 parameters
  width 0.5  -> [16, 32, 64, 128] ≈ 0.3 million
  width 1.0  -> [32, 64, 128, 256] ≈ 1.2 million
"fast": the first stage is only ONE Conv3×3 with stride 2 (instead of 2× Conv3×3 at full resolution + MaxPool).
  Same output size, but about half the compute - the first stage costs most of the time on the board.
The architecture is saved in the .pth, so that converting and testing recognise the variant automatically.
"""
import math

import torch
import torch.nn as nn

BASE = [32, 64, 128, 256]


def channels(width):
    return [max(16, int(math.ceil(c * width / 16.0)) * 16) for c in BASE]


class Stage(nn.Module):
    def __init__(self, c_in, c_out, split):
        super().__init__()
        layers = []
        for i in range(2):
            ci = c_in if i == 0 else c_out
            if split and i == 1:
                layers += [nn.Conv2d(ci, ci, 3, 1, 1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU(inplace=True),
                           nn.Conv2d(ci, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(ci, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        layers.append(nn.MaxPool2d(2))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class Stem(nn.Module):
    """Fast first stage: Conv3×3 stride 2 - halves the resolution right away."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class Classifier(nn.Module):
    def __init__(self, classes, width=0.25, split=False, outputs=None, fast=False):
        super().__init__()
        ks = channels(width)
        stages, c = [], 3
        for i, k in enumerate(ks):
            stages.append(Stem(c, k) if (fast and i == 0) else Stage(c, k, split))
            c = k
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, outputs or classes)
        self.architecture = {"classes": classes, "width": width, "split": split, "channels": ks,
                             "outputs": outputs or classes, "fast": fast}

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.head(torch.flatten(x, 1))


def parameters(m):
    return sum(p.numel() for p in m.parameters())


def compute(m, size):
    """Multiplications (MAC) per crop - a measure of the compute time on the ESP32 (conv + linear)."""
    total = [0]

    def count(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            total[0] += mod.in_features * mod.out_features
    hooks = [x.register_forward_hook(count) for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    was = m.training
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, size, size, device=next(m.parameters()).device))
    m.train(was)
    for h in hooks:
        h.remove()
    return total[0]


def load(path, device="cpu"):
    d = torch.load(path, map_location=device, weights_only=False)
    a = d["architecture"]
    m = Classifier(a["classes"], a["width"], a["split"], a.get("outputs"), a.get("fast", False))
    m.load_state_dict(d["model"])
    return m.to(device).eval(), d   # on the device of the data (otherwise: GPU data vs. CPU weights)


def padded(m, outputs=16):
    """Copy with the Linear padded to `outputs` (weights 0, bias -20 => softmax ≈ 0 for the extra outputs)."""
    import copy
    n = copy.deepcopy(m).eval()
    old = n.head
    if old.out_features >= outputs:
        return n
    new = nn.Linear(old.in_features, outputs)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:old.out_features] = old.weight
        new.bias.fill_(-20.0)
        new.bias[:old.out_features] = old.bias
    n.head = new
    n.architecture = dict(m.architecture, outputs=outputs)
    return n
