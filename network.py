import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import cv2
from torchvision.utils import make_grid
import numpy as np


class AMG_GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True, channels=1024):
        super().__init__()
        self.eps = eps
        self.channels = channels
        if learn_p:
            self.p = nn.Parameter(torch.ones(channels) * p)
        else:
            self.p = torch.ones(channels) * p

    def forward(self, x, mask):
        """
        x:    (B, C, H, W)
        mask: (B, 1, H, W), attention α(i,j), 0-1
        """
        mask = mask.clamp(min=self.eps)
        x = x.clamp(min=self.eps)
        p = self.p.view(1, -1, 1, 1)
        num = torch.sum(mask * (x ** p), dim=(2, 3))  # (B, C)
        den = torch.sum(mask, dim=(2, 3)) + self.eps  # (B, 1)
        out = (num / den) ** (1.0 / p.view(1, -1))
        return out


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=256, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()  # output attention map ∈ [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ------------------------ Main VPR Model ----------------------------
class HamVPRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_large(patch_size=14, img_size=518, init_values=1, block_chunks=0)
        self.attention_unet = AttentionUNet(in_channels=1024)
        self.amg_gem = AMG_GeM(p=3.0, eps=1e-6, learn_p=True, channels=1024)
        self.l2norm = L2Norm()

    def forward(self, image):
        with torch.no_grad():
            features = self.backbone(image)["x_norm_patchtokens"]
        patch_feature = features.view(-1, 16, 16, 1024)
        feat_map = patch_feature.permute(0, 3, 1, 2)
        pred_mask = self.attention_unet(feat_map)  # (B,1,16,16)
        pred_mask = pred_mask.clamp(1e-6, 1 - 1e-6)
        # AMG-GeM with mask
        global_feat = self.amg_gem(feat_map, pred_mask)
        global_feat = self.l2norm(global_feat)
        return global_feat
