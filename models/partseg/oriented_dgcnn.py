import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from modules.frame import FrameNetwork, HierFrameNetwork, MultiScaleFrameNetwork
from modules.geometric import global_to_local
from ._registry import register_model


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    """
    Args:
        x:  (B, d, N)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    if k is None: k = idx.size(-1)

    _, num_dims, _ = x.size()
    feature = gather(x, idx)
    x = x.transpose(2, 1).contiguous()
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B, d, N, K)
  
    return feature


def gather(x, idx):
    """
    Args:
        x:   (B, d, N)
        idx: (B, N, k)
    Returns:
        (B, N, K, d)
    """
    batch_size, num_dims, num_points = x.size()
    k = idx.size(2)
    assert x.size(2) == idx.size(1)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    return feature 


def get_projected_feature(p, R, k):
    """
    Args:
        p:  (B, 3, N)
    Returns:
        (B, 3, N, K)
    """
    idx = knn(p, k) # (B, N, k)
    p_j = gather(p, idx)  # (B, N, K, 3)
    p_ij = global_to_local(R, p.transpose(1,2).contiguous(), p_j)   # (B, N, K, 3)
    return p_ij.permute(0, 3, 1, 2).contiguous(), idx


@register_model('oriented_dgcnn')
class OrientedDGCNN(nn.Module):
    def __init__(self, cfg, seg_num_all):
        super().__init__()
        self.cfg = cfg
        self.seg_num_all = seg_num_all
        self.k = cfg.n_knn

        self.frame_net = MultiScaleFrameNetwork(
            hidden_dims = (cfg.frame.hidden_dim_s, cfg.frame.hidden_dim_v),
            num_layers = cfg.frame.num_layers,
            num_frames = cfg.frame.num_frames,
            k = cfg.frame.knn,
            scales = cfg.frame.scales,

            # scales = [2048, 256, 64, 1],
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(cfg.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        inp_dim = 3*(self.frame_net.total_frames + cfg.frame.num_frames*2)

        self.conv1 = nn.Sequential(nn.Conv2d(inp_dim, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, cfg.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),    # 16: Number of categories in the dataset
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=cfg.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=cfg.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, cat):
        """
        Args:
            x:  (B, N, 3).
        """
        batch_size = x.size(0)
        num_points = x.size(1)

        R, h, p_anchor, R_anchor = self.frame_net(x, return_anchors=True)
        p_center = p_anchor[-1].repeat(1, x.size(1), 1)     # (B, N, 3)
        R_center = R_anchor[-1].repeat(1, x.size(1), 1, 1)  # (B, N, F*3, 3)
        x_global = global_to_local(R_center, p_center, x)   # (B, N, F*3)


        x = x.permute(0, 2, 1).contiguous() # (B, N, 3) -> (B, 3, N)
        x_global = x_global.permute(0, 2, 1).contiguous()

        x, idx = get_projected_feature(x, R, k=self.k)
        x_glob_feat = get_graph_feature(x_global, k=None, idx=idx)

        x = torch.cat([x, x_glob_feat], dim=1)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = F.one_hot(cat.reshape(batch_size), 16).float()  # (batch_size, ) -> (batch_size, num_cats)
        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        x = x.transpose(1, 2).contiguous()
        return x

    def get_loss(self, x, seg, cat, smoothing=True):
        """Calculate cross entropy loss, apply label smoothing if needed. 
        Args:
            x:    Input point clouds, (B, N, 3).
            cat:  Ground truth point-wise labels, (B, N)
        """
        B, N, _ = x.size()

        out = self(x, cat)

        pred = out.reshape(B*N, self.seg_num_all)
        gold = seg.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss, out
