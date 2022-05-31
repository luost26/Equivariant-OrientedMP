import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.frame import MultiScaleFrameNetwork
from modules.geometric import global_to_local
from modules.rsconv import OrientedAnchoredRSConv
from ._registry import register_model


def get_hierarchical_idx(n, h=[512, 128, ]):
    all = np.arange(n)
    idx = []
    for m in h:
        # idx.append(np.random.choice(all, m, replace=False))
        idx.append(all[:m]) # Farthest point sampling
    return idx


@register_model('oriented_rscnn')
class OrientedRSCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.frame_net = MultiScaleFrameNetwork(
            hidden_dims = (cfg.frame.hidden_dim_s, cfg.frame.hidden_dim_v),
            num_layers = cfg.frame.num_layers,
            num_frames = cfg.frame.num_frames,
            k = cfg.frame.knn,
            scales=[1024,] + [512, 128, 1],
        )

        self.xyz_raising = nn.Sequential(
            nn.Conv1d(cfg.frame.num_frames*3, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.conv1 = OrientedAnchoredRSConv(32, 128, k=48, num_frames=cfg.frame.num_frames)
        self.conv2 = OrientedAnchoredRSConv(128, 512, k=64, num_frames=cfg.frame.num_frames)
        self.conv3 = OrientedAnchoredRSConv(512, 1024, k=128, num_frames=cfg.frame.num_frames)

        self.classifier = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Conv1d(256, cfg.num_classes, kernel_size=1, bias=True),
        )

    def forward(self, p):
        """
        Args:
            p:  (B, N, 3).
        """
        R, h, p_anchor, R_anchor = self.frame_net(p, return_anchors=True)

        p0 = p
        p1, p2, p3 = p_anchor   # (B, 512, 3), (B, 128, 3), (B, 1, 3)
        R1, R2, R3 = R_anchor

        p_center = p_anchor[-1].repeat(1, p.size(1), 1)     # (B, N, 3)
        R_center = R_anchor[-1].repeat(1, p.size(1), 1, 1)  # (B, N, F*3, 3)
        p_global = global_to_local(R_center, p_center, p)   # (B, N, F*3)

        h = self.xyz_raising(p_global.permute(0, 2, 1).contiguous())   # (B, 32, N)
        h = h.permute(0, 2, 1).contiguous()

        h = self.conv1(p0, p1, R1, h)   # (B, 512, h1)
        h = self.conv2(p1, p2, R2, h)   # (B, 128, h2)
        h = self.conv3(p2, p3, R3, h)   # (B, 1, h3)

        h = h.permute(0, 2, 1).contiguous()   # (B, h3, 1)
        out = self.classifier(h).squeeze(-1)  # (B, n_cls)

        return out

    def get_loss(self, p, cls, return_result=True):
        """
        Args:
            p:    (B, N, 3).
            cls:  (B, 1) or (B, ).
        """
        logp_pred = self(p)
        cls = cls.view([cls.size(0)])
        loss = F.cross_entropy(logp_pred, cls, reduction='mean')
        if return_result:
            return loss, logp_pred
        else:
            return loss
