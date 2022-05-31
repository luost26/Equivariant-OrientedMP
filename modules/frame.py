import torch
import torch.nn as nn
import numpy as np
from pytorch3d.ops.knn import knn_gather, knn_points

from .gvp import GVPConv, GVP, build_graph, tuple_sum
from .geometric import normalize_vector, construct_3d_basis_from_2_vectors


def get_hierarchical_idx(n, h, fps=False):
    all = np.arange(n)
    idx = []
    for m in h:
        if m == n:
            idx.append(all)
        else:
            if fps:
                idx.append(all[:m]) # Farthest point sampling
            else:
                idx.append(np.random.choice(all, m, replace=False))
    return idx


def nearest_unpool(p_ctx, p_query, x):
    """
    Args:
        p_ctx:    Contextual positions, (B, M, 3).
        p_query:  Query positions, (B, N, 3)
        x:        Value of contextual points, (B, M, F).
    """
    _, idx, _ = knn_points(p_query, p_ctx, K=1) # (B, N, 1)
    y = knn_gather(x, idx)  # (B, N, 1, F)
    y = y.squeeze(2)
    return y


class DistanceRBF(nn.Module):

    def __init__(self, num_channels=64, start=0.0, stop=2.0):
        super().__init__()
        stop = stop * 10
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        # Gaussian smearing parameters
        offset = torch.linspace(start, stop, num_channels-2)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist, dim):
        """
        Args:
            dist:   (N, *, 1, *)
        Returns:
            (N, *, num_channels, *)
        """
        assert dist.size()[dim] == 1
        dist = dist * 10
        offset_shape = [1] * len(dist.size())
        offset_shape[dim] = -1

        overflow_symb = (dist >= self.stop).float()  # (N, *, 1, *)
        underflow_symb = (dist < self.start).float() # (N, *, 1, *)
        y = dist - self.offset.view(*offset_shape)   # (N, *, dim-2, *)
        y = torch.exp(self.coeff * torch.pow(y, 2))  # (N, *, dim-2, *)
        return torch.cat([underflow_symb, y, overflow_symb], dim=dim)


class FrameNetwork(nn.Module):

    def __init__(self, hidden_dims, num_frames=1, num_gconvs=3, num_layers=3, k=24, dropout_rate=0.5):
        super().__init__()
        self.hid_s, self.hid_v = hidden_dims
        self.k = k
        self.num_frames = num_frames
        self.edge_s = 64
        self.dist_rbf = DistanceRBF(num_channels=self.edge_s)

        gconvs = []
        for _ in range(num_gconvs):
            gconvs.append(GVPConv(hidden_dims, hidden_dims, edge_dims=(self.edge_s, 1), n_layers=num_layers, dropout_rate=dropout_rate))
        self.gconvs = nn.ModuleList(gconvs)

        self.out_mlp = GVP(hidden_dims, out_dims=(self.hid_s, 2*num_frames), activations=(None, None))
        
    def forward(self, p):
        """
        Args:
            p:  Point coordinates, (B, N, 3).
        """
        B, N = p.shape[:2]
        edge_index, d_ij, dist = build_graph(p, p, k=self.k)  # (B, N, K), (B, N, K, 3), (B, N, K)
        edge_attr = (
            self.dist_rbf(dist.unsqueeze(-1), dim=-1),      # (B, N, K, 64)
            normalize_vector(d_ij, dim=-1).unsqueeze(-2),   # (B, N, K, 1, 3)    
        )
        x = (
            torch.zeros([B, N, self.hid_s]).to(p),
            torch.zeros([B, N, self.hid_v, 3]).to(p),
        )
        h = x
        for gconv in self.gconvs:
            h = gconv(h, edge_index, edge_attr)
        
        y_s, y_v = self.out_mlp(h)  # (B, N, hid_s), (B, N, F*2, 3)
        v = y_v.reshape(B, N, self.num_frames, 2, 3)
        R = construct_3d_basis_from_2_vectors(*torch.unbind(v, dim=-2)) # (B, N, F, 3, 3)
        R = R.reshape(B, N, self.num_frames*3, 3)
        return R, y_s


class HierFrameNetwork(nn.Module):

    def __init__(self, hidden_dims, hier=[1024, 256, 64, 256, 1024], num_frames=1,  num_layers=3, k=24, dropout_rate=0.5):
        super().__init__()
        self.hid_s, self.hid_v = hidden_dims
        self.hier = hier
        self.k = k
        self.num_frames = num_frames
        self.edge_s = 64
        self.dist_rbf = DistanceRBF(num_channels=self.edge_s)

        gconvs = []
        for _ in range(len(hier) - 1):
            gconvs.append(GVPConv(hidden_dims, hidden_dims, edge_dims=(self.edge_s, 1), n_layers=num_layers, dropout_rate=dropout_rate, shortcut=False))
        self.gconvs = nn.ModuleList(gconvs)

        self.out_mlp = GVP(hidden_dims, out_dims=(self.hid_s, 2*num_frames), activations=(None, None))

    def forward(self, p):
        """
        Args:
            p:  Point coordinates, (B, N, 3).
        """
        B, N = p.shape[:2]
        hier_idx = get_hierarchical_idx(p.size(1), self.hier, fps=not self.training)
        p_hier = [p[:, idx] for idx in hier_idx]
        x = (
            torch.zeros([B, N, self.hid_s]).to(p),
            torch.zeros([B, N, self.hid_v, 3]).to(p),
        )
        h = x
        for i, (p0, p1) in enumerate(zip(p_hier[:-1], p_hier[1:])):
            M = p1.size(1)
            edge_index, d_ij, dist = build_graph(p_ctx=p0, p_query=p1, k=self.k)  # (B, M, K), (B, M, K, 3), (B, M, K)
            edge_attr = (
                self.dist_rbf(dist.unsqueeze(-1), dim=-1),      # (B, N, K, 64)
                normalize_vector(d_ij, dim=-1).unsqueeze(-2),   # (B, N, K, 1, 3)    
            )
            h = self.gconvs[i](h, edge_index, edge_attr, 0)
        
        y_s, y_v = self.out_mlp(h)  # (B, N, hid_s), (B, N, F*2, 3)
        v = y_v.reshape(B, N, self.num_frames, 2, 3)
        R = construct_3d_basis_from_2_vectors(*torch.unbind(v, dim=-2)) # (B, N, F, 3, 3)
        R = R.reshape(B, N, self.num_frames*3, 3)
        return R, y_s


class MultiScaleFrameNetwork(nn.Module):

    def __init__(self, hidden_dims, scales=[1024, 256, 64, 1], num_gconvs=2, num_frames=4,  num_layers=3, k=24, dropout_rate=0.5):
        super().__init__()
        self.hid_s, self.hid_v = hidden_dims
        self.scales = scales
        self.k = k
        self.num_frames = num_frames
        self.edge_s = 64
        self.dist_rbf = DistanceRBF(num_channels=self.edge_s)

        gconvs_encode = []
        for _ in range(num_gconvs):
            gconvs_encode.append(GVPConv(hidden_dims, hidden_dims, edge_dims=(self.edge_s, 1), n_layers=num_layers, dropout_rate=dropout_rate))
        self.gconvs_encode = nn.ModuleList(gconvs_encode)

        gconvs_scale = []
        out_mlps = []
        for _ in range(len(scales)-1):
            gconvs_scale.append(GVPConv(hidden_dims, hidden_dims, edge_dims=(self.edge_s, 1), n_layers=num_layers, dropout_rate=dropout_rate, shortcut=False))
            out_mlps.append(GVP(hidden_dims, out_dims=(self.hid_s, 2*num_frames), activations=(None, None)))
        self.gconvs_scale = nn.ModuleList(gconvs_scale)
        self.out_mlps = nn.ModuleList(out_mlps)

        self.out_mlp = GVP(hidden_dims, out_dims=(self.hid_s, 2*num_frames), activations=(None, None))

    @property
    def total_frames(self):
        return self.num_frames * (len(self.scales)-1)

    def forward(self, p, return_anchors=False):
        """
        Args:
            p:  Point coordinates, (B, N, 3).
        """
        B, N = p.shape[:2]

        # Build hierarchy
        hier_idx = get_hierarchical_idx(p.size(1), self.scales, fps=not self.training)
        p_hier = [p[:, idx] for idx in hier_idx]
        if self.scales[-1] == 1:
            # p_hier[-1] = (p.max(dim=1, keepdim=True)[0] + p.min(dim=1, keepdim=True)[0]) / 2  # WARNING: This way violates rotational invariance
            center = p.mean(dim=1, keepdim=True)
            if self.training:
                center = center + torch.randn_like(center) * p.std() / 4
            p_hier[-1] = center
            

        # Encode points
        edge_index, d_ij, dist = build_graph(p, p, k=self.k)  # (B, N, K), (B, N, K, 3), (B, N, K)
        edge_attr = (
            self.dist_rbf(dist.unsqueeze(-1), dim=-1),      # (B, N, K, 64)
            normalize_vector(d_ij, dim=-1).unsqueeze(-2),   # (B, N, K, 1, 3)    
        )
        x = (torch.zeros([B, N, self.hid_s]).to(p), torch.zeros([B, N, self.hid_v, 3]).to(p))
        h = x
        for gconv in self.gconvs_encode:
            h = gconv(h, edge_index, edge_attr)
        
        feat_point = h[0].clone()

        R_all, R_anchor = [], []
        for p_prev, p_this, gconv, out_net in zip(p_hier[:-1], p_hier[1:], self.gconvs_scale, self.out_mlps):
            M = p_this.size(1)
            edge_index, d_ij, dist = build_graph(p_ctx=p_prev, p_query=p_this, k=self.k)  # (B, M, K), (B, M, K, 3), (B, M, K)
            edge_attr = (
                self.dist_rbf(dist.unsqueeze(-1), dim=-1),      # (B, M, K, 64)
                normalize_vector(d_ij, dim=-1).unsqueeze(-2),   # (B, M, K, 1, 3)    
            )
            h = gconv(h, edge_index, edge_attr, 0)  # (B, M, [])

            _, y_v = out_net(h)  # (B, M, hid_s), (B, M, F*2, 3)
            v = y_v.reshape(B, M, self.num_frames, 2, 3)
            R = construct_3d_basis_from_2_vectors(*torch.unbind(v, dim=-2)) # (B, M, F, 3, 3)
            
            R_anchor.append(R.reshape(B, M, self.num_frames*3, 3))
            
            R = R.reshape(B, M, self.num_frames*3*3)   # (B, M, F*3*3)
            R = nearest_unpool(p_this, p, R).reshape(B, N, self.num_frames*3, 3)    # (B, N, F*3, 3)
            R_all.append(R)

        R_all = torch.cat(R_all, dim=2) # (B, N, S*F*3, 3)

        if return_anchors:
            return R_all, feat_point, p_hier[1:],  R_anchor
        else:
            return R_all, feat_point
