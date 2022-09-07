# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Source image encoder for ViewSeg.
"""
import math 
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchvision
from torchvision import transforms as T

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.projects.point_rend import add_pointrend_config

from .backbone import build_backbone


def maybe_permute(images):
    if images.ndim < 4:
        images = images[None, ...]

    if images.ndim == 4:
        if images.shape[1] != 3:
            # (B, H, W, C) -> (B, C, H, W)
            images = images.permute(0, 3, 1, 2)
    elif images.ndim == 5:
        if images.shape[2] != 3:
            # (B, NV, H, W, C) -> (B, NV, C, H, W)
            images = images.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError("Images dimensions weird {}".format(images.ndim))
        
    return images


def add_hypersim_config(cfg):
    # ade20k pretraining
    cfg.MODEL.META_ARCHITECTURE = "SemanticSegmentor"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.SEM_SEG_HEAD.NAME = "DeepLabV3Head"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res5"]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 16
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 150
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    cfg.MODEL.SEM_SEG_HEAD.SIZE_DIVISIBILITY = 512
    cfg.INPUT.MIN_SIZE_TRAIN: [int(x * 0.1 * 512) for x in range(5, 21)]
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (512, 512)
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.COLOR_AUG_SSD = True
    cfg.INPUT.FORMAT = "RGB"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 10

    # deeplabv3+
    cfg.MODEL.WEIGHTS = "detectron2://DeepLab/R-52.pkl"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.BACKBONE.NAME = "build_resnet_deeplab_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res5"]
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.SEM_SEG_HEAD.NAME = "DeepLabV3PlusHead"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.NORM = "SyncBN"
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4

    # disable aspp pooling
    cfg.INPUT.CROP.ENABLED = False

    return cfg


class ResNetEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        norm_type="batch",
        use_first_pool=True,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        model_path=None,
    ):
        """
        Args:
            # Backbone params
            backbone: Backbone network. e.g. resnet18/resnet34 model from torchvision
            pretrained: Whether to use model weights pretrained on ImageNet
            num_layers: number of resnet layers to use, 1-5
            norm_type: norm type to applied; pretrained model must use batch
            use_first_pool: if false, skips first maxpool layer to avoid downscaling image
                features too much (ResNet only)
            # Params for feature sampling 
            index_interp Interpolation to use for indexing
            :param index_padding Padding mode to use for indexing, border | zeros | reflection
            :param upsample_interp Interpolation to use for upscaling latent code
            :param feature_scale factor to scale all latent by. Useful (<1) if image
            is extremely large, to fit in memory.
            :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
            features too much (ResNet only)
        """
        super().__init__()

        # Normalization required by torchvision models 
        self.normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

        if norm_type != "batch":
            assert not pretrained

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool

        self.model, self.latent_size = build_backbone(
            backbone, pretrained, norm_type, num_layers
        )

        # Following 2 lines need to be uncommented for older configs
        # self.model.fc = nn.Sequential()
        # self.model.avgpool = nn.Sequential()
        # self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        
        # Latent vector = (B, D, H, W)
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, D, H, W)

    def process_image(self, images):
        """
        For a batch of images, reshape dims and apply transformation
        required by torchvision
        """
        images = maybe_permute(images)
        
        assert images.max() <= 1.0 and images.min() >= 0.0

        # Apply image transform required for torchvision models
        images = self.normalize_transform(images)
        return images

    def forward(self, x):
        """
        Extract ResNet features.
        
        Args:
            x: image (B, H, W, 3)
        Return:
            latent: (B, self.latent_size, H, W)
        """
        # Reshape dims and apply transform to pixel values 
        N = x.shape[0]
        if x.ndim == 5:
            num_views = x.shape[1]
            x = x.reshape(N*num_views, *x.shape[2:])
        x = self.process_image(x)

        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        self.latents = latents
        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0

        return self.latent


class DeepLabv3PlusEncoder(nn.Module):
    """
    Deeplab v3+ decoder
    """

    def __init__(
        self,
        backbone="deeplabv3_plus",
        bn='SyncBN',
        pretrained=True,
        num_layers=4,
        norm_type="batch",
        use_first_pool=True,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        model_path=None,
    ):
        """
        Args:
            # Backbone params
            backbone: Backbone network. e.g. resnet18/resnet34 model from torchvision
            pretrained: Whether to use model weights pretrained on ImageNet
            num_layers: number of resnet layers to use, 1-5
            norm_type: norm type to applied; pretrained model must use batch
            use_first_pool: if false, skips first maxpool layer to avoid downscaling image
                features too much (ResNet only)
            # Params for feature sampling 
            index_interp Interpolation to use for indexing
            :param index_padding Padding mode to use for indexing, border | zeros | reflection
            :param upsample_interp Interpolation to use for upscaling latent code
            :param feature_scale factor to scale all latent by. Useful (<1) if image
            is extremely large, to fit in memory.
            :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
            features too much (ResNet only)
        """
        super().__init__()

        # this is the same as detectron2 normalization
        # PIXEL_MEAN: [123.675, 116.280, 103.530]
        # PIXEL_STD: [58.395, 57.120, 57.375]
        self.normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

        self.feature_scale = feature_scale

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_pointrend_config(cfg)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        cfg.merge_from_file(model_zoo.get_config_file('Base-RCNN-DilatedC5.yaml'))
        add_hypersim_config(cfg)

        if pretrained:
            print("Deeplabv3 is using ADE20k pretraining. Ignore this if you do not train from scratch.")
            cfg.MODEL.WEIGHTS = model_path
        else:
            print("Deeplabv3 is using ImageNet pretraining. Ignore this if you do not train from scratch.")
    
        # change BN type of deeplabv3 backbone
        if bn not in ['SyncBN', 'BN', 'FrozenBN']:
            raise ValueError(bn)
        else:
            cfg['MODEL']['RESNETS']['NORM'] = bn
            cfg['MODEL']['SEM_SEG_HEAD']['NORM'] = bn

        # load model
        self.model = DefaultPredictor(cfg).model

        self.latent_size = 256

    def process_image(self, images):
        """
        For a batch of images, reshape dims and apply transformation
        required by torchvision
        """
        images = maybe_permute(images)
        
        assert images.max() <= 1.0 and images.min() >= 0.0

        # Apply image transform required for torchvision models
        images = self.normalize_transform(images)
        return images

    def forward(self, x):
        """
        Extract Deeplabv3+ features.
        
        Args:
            x: image (B, H, W, 3)
        Return:
            latent: (B, self.latent_size, H, W)
        """
        # Reshape dims and apply transform to pixel values 
        N = x.shape[0]
        if x.ndim == 5:
            num_views = x.shape[1]
            x = x.reshape(N*num_views, *x.shape[2:])
        x = self.process_image(x)

        # deeplabv3+ except the final layer 'predictor'
        features = self.model.backbone(x)
        self.model.sem_seg_head.decoder_only = True # no predictor
        self.latent = self.model.sem_seg_head.layers(features)

        return self.latent


def build_spatial_encoder(
    backbone="resnet34",
    bn="SyncBN",
    pretrained=True,
    num_layers=4,
    norm_type="batch",
    use_first_pool=True,
    index_interp="bilinear",
    index_padding="border",
    upsample_interp="bilinear",
    feature_scale=1.0,
    model_path=None,
):
    if backbone.startswith("resnet"):
        model = ResNetEncoder(
            backbone,
            pretrained,
            num_layers,
            norm_type,
            use_first_pool,
            index_interp,
            index_padding,
            upsample_interp,
            feature_scale,
            model_path,
        )
    elif backbone.startswith("deeplabv3"):
        model = DeepLabv3PlusEncoder(
            backbone,
            bn,
            pretrained,
            num_layers,
            norm_type,
            use_first_pool,
            index_interp,
            index_padding,
            upsample_interp,
            feature_scale,
            model_path,
        )
    
    return model
