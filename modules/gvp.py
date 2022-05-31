"""
GVP layers (not using pytorch geometric)
Adapted from:
    https://github.com/drorlab/gvp-pytorch/blob/main/gvp/__init__.py
"""
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)


def tuple_knn_gather(x, idx):
    nv = x[1].size(-2)
    x = _merge(*x)
    y = knn_gather(x, idx)
    return _split(y, nv)


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims, enable_scalar_norm=True):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        if enable_scalar_norm:
            self.scalar_norm = nn.LayerNorm(self.s)
        else:
            self.scalar_norm = nn.Identity()
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(nn.Module):

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=True, 
                 dropout_rate=0.5, shortcut=True):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        assert aggr in ('mean', 'sum')
        self.aggr = aggr
        self.shortcut = shortcut
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

        if self.si == self.so and self.vi == self.vo:
            self.shortcut_transform = nn.Identity()
        else:
            self.shortcut_transform = GVP_(in_dims, out_dims, activations=(None, None))

        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNorm(out_dims, enable_scalar_norm=False)

    def forward(self, x, edge_index, edge_attr, x_last = None):
        """
        Args:
            x:          Pointwise feature to query from, tuple [(B, N, si), (B, N, vi, 3)].
            x_last:     Pointwise feature to be updated, tuple [(B, M, si), (B, M, vi, 3)].
            edge_index: Node index of k-nearest neighbors, (B, M, K).
            edge_attr:  Edge feature, tuple [(B, M, K, se), (B, M, K, ve, 3)].
        """
        B, M, K = edge_index.size()
        if x_last is None: 
            x_last = x
        elif x_last == 0:
            x_last = (
                torch.zeros([B, M, self.si]).to(x[0]),
                torch.zeros([B, M, self.vi, 3]).to(x[1]),
            )
        x_i = (
            x_last[0].unsqueeze(2).repeat(1, 1, K, 1),     # (B, M, K, si) 
            x_last[1].unsqueeze(2).repeat(1, 1, K, 1, 1),  # (B, M, K, vi, 3)  
        )
        x_j = tuple_knn_gather(x, edge_index)   # (B, M, K, [si/vi3])
        # print('x_i', x_i[0].size(), x_last[0].size())
        # print('x_j', x_j[0].size(), x[0].size())
        inp = tuple_cat(x_i, x_j, edge_attr)
        message = self.message_func(inp)    # (B, M, K, [])

        if self.aggr == 'mean':
            dx = (message[0].mean(dim=2), message[1].mean(dim=2))
        elif self.aggr == 'sum':
            dx = (message[0].sum(dim=2), message[1].sum(dim=2)) # (B, M, [])

        if self.shortcut:
            out = tuple_sum(self.shortcut_transform(x_last), self.dropout(dx))
        else:
            out = self.dropout(dx)
        # out = self.norm(out)
        return out


def build_graph(p_ctx, p_query, k):
    """
    Args:
        p_ctx:      Pointwise 3D coordinates, (B, N, 3).
        p_query:    (B, M, 3)
    Returns:
        (B, M, K)
    """
    dist2, idx, p_j = knn_points(p_query, p_ctx, K=k, return_nn=True) # (B, M, K), (B, M, K), (B, M, K, 3)
    dist = dist2.sqrt()
    p_i = p_query.unsqueeze(2).repeat(1, 1, k, 1) # (B, N, K, 3)
    d_ij = p_j - p_i    # (B, N, K, 3)
    return idx, d_ij, dist
