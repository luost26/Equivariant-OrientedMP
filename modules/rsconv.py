import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points, knn_gather

from .geometric import global_to_local


class RSConv(nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.in_channels = in_channels
        self.out_channales = out_channels
        self.k = k

        self.weight_network = nn.Sequential(
            nn.Conv2d(10, in_channels//4, kernel_size=(1, 1)), 
            nn.BatchNorm2d(in_channels//4), 
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels, kernel_size=(1, 1)),
        )
        
        self.conv_bn = nn.BatchNorm2d(in_channels)
        self.conv_act = nn.ReLU()

        self.out_network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )


    def forward(self, p_in, p_out, h_in):
        """
        Args:
            p_in:   (B, N_in, 3)
            p_out:  (B, N_out, 3)
            h_in:   (B, N_in, in_ch)
        Returns:
            h_out:  (B, N_out, out_ch)
        """
        _, idx, p_j = knn_points(p_out, p_in, K=self.k, return_nn=True)   # (B, N_out, K), (B, N_out, K), (B, N_out, K, 3)
        p_i = p_out.unsqueeze(2).repeat(1, 1, self.k, 1) # (B, N_out, K, 3)
        p_ij = (p_j - p_i)  # (B, N_out, K, 3)
        d_ij = torch.linalg.norm(p_ij, dim=-1, keepdim=True)    # (B, N_out, K, 1)

        w_ij = torch.cat([p_ij, d_ij, p_i, p_j], dim=-1)    # (B, N_out, K, 3+3+3+1)
        w_ij = self.weight_network(w_ij.permute(0, 3, 1, 2).contiguous())   # (B, in_ch, N_out, K)

        h_j = knn_gather(h_in, idx).permute(0, 3, 1, 2).contiguous()    # (B, N_out, K, in_ch) -> (B, in_ch, N_out, K)
        m_ij = self.conv_act(self.conv_bn(w_ij * h_j))  # (B, in_ch, N_out, K)

        h_out = m_ij.max(dim=-1)[0] # (B, in_ch, N_out)
        h_out = self.out_network(h_out).permute(0, 2, 1).contiguous()   # (B, out_ch, N_out) -> (B, N_out, out_ch)
        return h_out


class OrientedAnchoredRSConv(nn.Module):

    def __init__(self, in_channels, out_channels, k, num_frames):
        super().__init__()
        self.in_channels = in_channels
        self.out_channales = out_channels
        self.k = k
        self.num_frames = num_frames

        self.weight_network = nn.Sequential(
            nn.Conv2d(num_frames*4, in_channels//4, kernel_size=(1, 1)), 
            nn.BatchNorm2d(in_channels//4), 
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels, kernel_size=(1, 1)),
        )
        
        self.conv_bn = nn.BatchNorm2d(in_channels)
        self.conv_act = nn.ReLU()

        self.out_network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )


    def forward(self, p_in, p_out, R_out, h_in):
        """
        Args:
            p_in:   (B, N_in, 3)
            p_out:  (B, N_out, 3)
            R_out:  (B, N_out, F*3, 3)
            h_in:   (B, N_in, in_ch)
        Returns:
            h_out:  (B, N_out, out_ch)
        """
        B, N_in, N_out = p_in.size(0), p_in.size(1), p_out.size(1)

        _, idx, p_j = knn_points(p_out, p_in, K=self.k, return_nn=True)   # (B, N_out, K), (B, N_out, K), (B, N_out, K, 3)

        p_ij = global_to_local(R_out, p_out, p_j)   # (B, N_out, K, F*3)
        d_ij = torch.linalg.norm(p_ij.reshape(B, N_out, self.k, self.num_frames, 3), dim=-1, keepdim=False)    # (B, N_out, K, F)

        w_ij = torch.cat([p_ij, d_ij], dim=-1)    # (B, N_out, K, 3+1)
        w_ij = self.weight_network(w_ij.permute(0, 3, 1, 2).contiguous())   # (B, in_ch, N_out, K)

        h_j = knn_gather(h_in, idx).permute(0, 3, 1, 2).contiguous()    # (B, N_out, K, in_ch) -> (B, in_ch, N_out, K)
        m_ij = self.conv_act(self.conv_bn(w_ij * h_j))  # (B, in_ch, N_out, K)

        h_out = m_ij.max(dim=-1)[0] # (B, in_ch, N_out)
        h_out = self.out_network(h_out).permute(0, 2, 1).contiguous()   # (B, out_ch, N_out) -> (B, N_out, out_ch)
        return h_out
