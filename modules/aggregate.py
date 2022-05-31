import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points, knn_gather

from .geometric import global_to_local


class PointOrientedAggregation(nn.Module):
    
    def __init__(self, hidden_dim, feat_net_type, num_frames=1, graph_type='hidden', k=24, aggr='max'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k

        assert feat_net_type in ('mlp', 'linear', 'perceptron')
        in_dim = hidden_dim*2 + 3*num_frames
        if feat_net_type == 'mlp':
            self.feat_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif feat_net_type == 'linear':
            self.feat_net = nn.Linear(in_dim, hidden_dim)
        elif feat_net_type == 'perceptron':
            self.feat_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.1),
            )
        
        assert aggr in ('max', 'mean', 'max')
        self.aggr = aggr

        assert graph_type in ('pos', 'hidden')
        self.graph_type = graph_type

    def forward(self, p, R, h):
        """
        Args:
            p:  Point coordinates, (B, N, 3).
            R:  Local frames, (B, N, 3, 3).
            h:  Point features, (B, N, h).
        """
        if self.graph_type == 'pos':
            _, idx, _ = knn_points(p, p, K=self.k)   # (B, N, K)
        elif self.graph_type == 'hidden':
            _, idx, _ = knn_points(h, h, K=self.k)
        p_j = knn_gather(p, idx)
        p_ij = global_to_local(R, p, p_j)   # (B, N, K, 3)
        h_j = knn_gather(h, idx)    # (B, N, K, h)
        h_i = h.unsqueeze(2).repeat(1, 1, self.k, 1) # (B, N, K, h)

        feat_ij = torch.cat([p_ij, h_i, h_j], dim=-1) # (B, N, K, h*2+3)
        # print(feat_ij.size())
        feat_ij = self.feat_net(feat_ij)

        if self.aggr == 'max': out = feat_ij.max(dim=2)[0]
        elif self.aggr == 'mean': out = feat_ij.mean(dim=2)
        elif self.aggr == 'sum': out = feat_ij.sum(dim=2)

        return out
