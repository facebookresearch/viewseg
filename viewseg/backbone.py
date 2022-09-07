# Copyright (c) Meta Platforms, Inc. and affiliates.
import functools
import torchvision
from torch import nn
import torch.nn.functional as F

def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


# Convention for feat dims is following how the num_layers is defined in pixel_nerf
# https://github.com/sxyu/pixel-nerf/blob/2929708e90b246dbd0329ce2a128ef381bd8c25d/src/model/encoder.py#L68
_FEAT_DIMS = {
    "resnet18": (0, 64, 128, 256, 512),
    "resnet34": (0, 64, 128, 256, 512),
    "resnet50": (0, 256, 512, 1024, 2048),
    "resnet101": (0, 256, 512, 1024, 2048),
    "resnet152": (0, 256, 512, 1024, 2048),
}


def build_backbone(name, pretrained=True, norm_type="batch", num_layers=4):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if name in resnets and name in _FEAT_DIMS:
        norm_layer = get_norm_layer(norm_type)
        cnn = getattr(torchvision.models, name)

        backbone = cnn(pretrained=pretrained, norm_layer=norm_layer)
        feat_dims = _FEAT_DIMS[name]
        assert num_layers < len(feat_dims)
        latent_feat_dim = feat_dims[num_layers]
        return backbone, latent_feat_dim
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)
