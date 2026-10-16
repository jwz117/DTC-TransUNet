"""Building blocks used by DTC-TransUNet."""
import math
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 2) Core blocks
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# -----------------------------
# 3) Morphological Prior Extractor (MPE)
# -----------------------------

class DWStripConv(nn.Module):
    """Depth-wise strip convolution: kx1 or 1xk (groups=channels)"""

    def __init__(self, channels: int, k: int, orientation: str = "h"):
        super().__init__()
        assert orientation in ("h", "v")
        if orientation == "h":  # 1 x k
            kernel = (1, k)
            padding = (0, k // 2)
        else:  # k x 1
            kernel = (k, 1)
            padding = (k // 2, 0)
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=kernel, padding=padding, groups=channels, bias=False
        )

    def forward(self, x):
        return self.dw(x)


class BoundaryDetectionModuleBC(nn.Module):
    """Boundary module (prob + confidence)."""

    def __init__(self, channels: int, k1: int = 3, k2: int = 5):
        super().__init__()
        self.h1 = DWStripConv(channels, k1, "h")
        self.v1 = DWStripConv(channels, k1, "v")
        self.g1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

        self.h2 = DWStripConv(channels, k2, "h")
        self.v2 = DWStripConv(channels, k2, "v")
        self.g2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.prob_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b1 = self.h1(x) + self.v1(x)
        b1 = b1 * self.g1(b1)

        b2 = self.h2(x) + self.v2(x)
        b2 = b2 * self.g2(b2)

        y = b1 + b2
        y = self.pw(self.dw7(y))

        edge_prob = self.prob_head(y)
        edge_conf = self.conf_head(y)
        return edge_prob, edge_conf


class StructurePriorModule(nn.Module):
    """Structure prior (prob + confidence)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.prob_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.block(x)
        s_prob = self.prob_head(h)
        s_conf = self.conf_head(h)
        return s_prob, s_conf


class MorphologicalPriorExtractor(nn.Module):
    """Semantic-aware BC morphological priors.

    Inputs:
      x_img: [B, Cin, H, W]
      feat2:  encoder mid feature (e.g. s2)
      feat3:  encoder deep feature (e.g. b)
    Outputs:
      e_morph, s_morph: [B,1,H,W]
    """

    def __init__(self, in_img: int, c2: int, c3: int, mid_channels: int = 32):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Conv2d(in_img, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2_proj = nn.Sequential(
            nn.Conv2d(c2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.enc3_proj = nn.Sequential(
            nn.Conv2d(c3, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.sem_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ctx_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.boundary_bc = BoundaryDetectionModuleBC(mid_channels, k1=3, k2=5)
        self.struct_mod = StructurePriorModule(mid_channels)

    def forward(self, x_img: torch.Tensor, feat2: torch.Tensor, feat3: torch.Tensor):
        H, W = x_img.shape[-2:]

        img_f = self.img_proj(x_img)

        f2 = F.interpolate(feat2, size=(H, W), mode="bilinear", align_corners=False)
        f3 = F.interpolate(feat3, size=(H, W), mode="bilinear", align_corners=False)
        f2 = self.enc2_proj(f2)
        f3 = self.enc3_proj(f3)
        sem_f = self.sem_fuse(torch.cat([f2, f3], dim=1))

        ctx = self.ctx_fuse(torch.cat([img_f, sem_f], dim=1))

        e_prob, e_conf = self.boundary_bc(ctx)
        s_prob, s_conf = self.struct_mod(sem_f + ctx)

        e_morph = (e_prob * e_conf).clamp(0.0, 1.0)
        s_morph = (s_prob * s_conf).clamp(0.0, 1.0)
        return e_morph, s_morph


class DynamicGatingModulation(nn.Module):
    """Dual Morphological Gate: feat*(1+αe)*(1+βs) then 1x1 proj."""

    def __init__(self, c: int):
        super().__init__()
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, feat: torch.Tensor, e_morph: torch.Tensor, s_morph: torch.Tensor, alpha=0.7, beta=0.5):
        if e_morph.shape[-2:] != feat.shape[-2:]:
            e_morph = F.interpolate(e_morph, feat.shape[-2:], mode="bilinear", align_corners=False)
        if s_morph.shape[-2:] != feat.shape[-2:]:
            s_morph = F.interpolate(s_morph, feat.shape[-2:], mode="bilinear", align_corners=False)
        x = feat * (1.0 + float(alpha) * e_morph)
        x = x * (1.0 + float(beta) * s_morph)
        return self.proj(x)


# -----------------------------
# 4) Transformer bottleneck (TransUNet-like)
# -----------------------------

class BottleneckTransformer(nn.Module):
    """b=[B,C,H,W] -> tokens [B,N,C] -> Transformer -> [B,C,H,W]"""

    def __init__(self, dim: int, depth: int = 4, nhead: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, b):
        B, C, H, W = b.shape
        x = b.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.encoder(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# -----------------------------
# 5) Dynamic Tri-source Consensus Graph Convolutional Network (DTC-GCN)
# -----------------------------
class LocalCNNBranch(nn.Module):
    """
    Parallel local CNN branch used to preserve local regular-grid information.
    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H, W]
    """
    def __init__(self, in_channels: int, mid_channels: int = 128, out_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DynamicTriSourceConsensusGCN(nn.Module):
    """
    Dynamic tri-source consensus graph reasoning with sample-specific node assignment and adjacency-constrained propagation.

    K-means node assignment, Sobel-based adjacency construction, and adjacency-constrained
    tri-source attention are implemented with PyTorch operations.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        compress_ratio: float = 0.5,
        gnn_layers: int = 2,
        T: int = 3,
        alpha_fuse: float = 0.5,
        nodes_per_pix: int = 512,
        max_nodes: int = 400,
        min_nodes: int = 16,
        kmeans_iters: int = 5,
        lambda_spatial: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(in_channels)
        self.compress_ratio = float(compress_ratio)

        feat_dim = max(8, int(round(self.in_channels * self.compress_ratio)))
        self.feat_dim = int(feat_dim)

        self.gnn_layers_n = int(gnn_layers)
        self.T = int(T)
        self.alpha_fuse = float(alpha_fuse)
        self.nodes_per_pix = int(nodes_per_pix)
        self.max_nodes = int(max_nodes)
        self.min_nodes = int(min_nodes)
        self.kmeans_iters = int(kmeans_iters)
        self.lambda_spatial = float(lambda_spatial)
        self.dropout = float(dropout)

        # Channel compression before graph construction.
        self.channel_compress = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )

        # Semantic, context, and spatial attention projections.
        self.q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.k_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.ctx_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.spatial_proj = nn.Linear(self.feat_dim, self.out_channels, bias=False)

        # dense message passing layers
        self.gnn_layers = nn.ModuleList([nn.Linear(self.feat_dim, self.feat_dim, bias=False)
                                         for _ in range(self.gnn_layers_n)])
        self.norm = nn.LayerNorm(self.feat_dim)

        # Parallel local CNN branch
        self.local_cnn_branch = LocalCNNBranch(self.feat_dim, mid_channels=128, out_channels=self.out_channels)

        # alpha fuse as buffer (fixed)
        self.register_buffer("_alpha_buf", torch.tensor(self.alpha_fuse, dtype=torch.float32), persistent=False)

        # Sobel kernels as buffers (GPU)
        gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("_sobel_gx", gx, persistent=False)
        self.register_buffer("_sobel_gy", gy, persistent=False)

        # coords cache (keyed by (H,W,device))
        self._coords_cache = {}

    @torch.no_grad()
    def _get_coords(self, H: int, W: int, device, dtype) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key in self._coords_cache:
            coords = self._coords_cache[key]
            if coords.device != device:
                coords = coords.to(device)
                self._coords_cache[key] = coords
            return coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        xx = xx / max(1.0, float(W))
        yy = yy / max(1.0, float(H))
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [HW,2]
        self._coords_cache[key] = coords
        return coords

    @torch.no_grad()
    def _edge_strength_map_torch(self, x_c_single: torch.Tensor) -> torch.Tensor:
        """
        x_c_single: [C,H,W] (on GPU)
        return: edge magnitude [H,W] in [0,1]
        """
        C, H, W = x_c_single.shape
        ch = min(3, C)
        img = x_c_single[:ch].mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        img_min = img.amin(dim=(-2, -1), keepdim=True)
        img_max = img.amax(dim=(-2, -1), keepdim=True)
        img = (img - img_min) / (img_max - img_min + 1e-6)

        gx = F.conv2d(img, self._sobel_gx.to(img.dtype), padding=1)
        gy = F.conv2d(img, self._sobel_gy.to(img.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy).squeeze(0).squeeze(0)  # [H,W]
        mag_min = mag.amin()
        mag_max = mag.amax()
        mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
        return mag

    @torch.no_grad()
    def _kmeans_torch(self, feat: torch.Tensor, K: int) -> torch.Tensor:
        """
        feat: [N, D] on GPU
        return labels: [N] long
        """
        N, D = feat.shape
        K = int(K)
        if K <= 1:
            return torch.zeros((N,), device=feat.device, dtype=torch.long)

        # init centers: random samples
        idx = torch.randperm(N, device=feat.device)[:K]
        centers = feat[idx].clone()  # [K,D]

        for _ in range(self.kmeans_iters):
            # assign
            dist = torch.cdist(feat, centers, p=2)  # [N,K]
            labels = torch.argmin(dist, dim=1)      # [N]

            # update
            counts_int = torch.bincount(labels, minlength=K)  # [K] (int)
            empty = counts_int == 0

            counts = counts_int.clamp_min(1).to(feat.dtype)   # [K] (float)
            sums = torch.zeros((K, D), device=feat.device, dtype=feat.dtype)
            sums.index_add_(0, labels, feat)
            new_centers = sums / counts[:, None]

            # re-init empty clusters (rare when K small)
            if empty.any():
                ridx = torch.randperm(N, device=feat.device)[:int(empty.sum().item())]
                new_centers[empty] = feat[ridx]

            centers = new_centers

        return labels

    @torch.no_grad()
    def _build_adj_mask(self, labels_hw: torch.Tensor, edge_mag: torch.Tensor, K: int) -> torch.Tensor:
        """
        labels_hw: [H,W] long
        edge_mag : [H,W] float
        K        : number of nodes
        return adj_mask: [K,K] float in {0,1}
        """
        device = labels_hw.device
        H, W = labels_hw.shape
        K = int(K)

        # down neighbors
        cur_d = labels_hw[:-1, :]
        nb_d = labels_hw[1:, :]
        mask_d = cur_d != nb_d
        if mask_d.any():
            i_d = cur_d[mask_d]
            j_d = nb_d[mask_d]
            w_d = 0.5 * (edge_mag[:-1, :][mask_d] + edge_mag[1:, :][mask_d])
        else:
            i_d = j_d = w_d = None

        # right neighbors
        cur_r = labels_hw[:, :-1]
        nb_r = labels_hw[:, 1:]
        mask_r = cur_r != nb_r
        if mask_r.any():
            i_r = cur_r[mask_r]
            j_r = nb_r[mask_r]
            w_r = 0.5 * (edge_mag[:, :-1][mask_r] + edge_mag[:, 1:][mask_r])
        else:
            i_r = j_r = w_r = None

        if i_d is None and i_r is None:
            adj_mask = torch.eye(K, device=device, dtype=torch.float32)
            return adj_mask

        if i_d is None:
            i = i_r; j = j_r; w = w_r
        elif i_r is None:
            i = i_d; j = j_d; w = w_d
        else:
            i = torch.cat([i_d, i_r], dim=0)
            j = torch.cat([j_d, j_r], dim=0)
            w = torch.cat([w_d, w_r], dim=0)

        # symmetrize
        idx1 = i * K + j
        idx2 = j * K + i
        idx = torch.cat([idx1, idx2], dim=0).long()
        w_all = torch.cat([w, w], dim=0).to(torch.float32)

        flat = torch.zeros((K * K,), device=device, dtype=torch.float32)

        # Prefer scatter_reduce_ (GPU, fast)
        if hasattr(flat, "scatter_reduce_"):
            flat.scatter_reduce_(0, idx, w_all, reduce="amax", include_self=True)
        else:
            # small-K fallback (still GPU tensors, but Python loop)
            # K is typically small (default min_nodes=16, so K*K=256)
            uniq = torch.unique(idx)
            for u in uniq.tolist():
                sel = (idx == u)
                flat[u] = torch.max(w_all[sel])

        # build binary mask + self loops
        adj = flat.view(K, K)
        adj_mask = (adj > 0).to(torch.float32)
        adj_mask.fill_diagonal_(1.0)
        return adj_mask

    @torch.no_grad()
    def build_spatial_nodes(self, x_c: torch.Tensor):
        """
        x_c: [B, feat_dim, H, W] (GPU)
        returns:
          node_feats_list: list of [K, feat_dim]
          adj_masks_list : list of [K, K] float
          labels_list    : list of [H, W] long
          centroids_list : list of [K, 2] normalized (x, y) node centroids
        """
        B, C, H, W = x_c.shape
        device = x_c.device
        dtype = x_c.dtype
        coords = self._get_coords(H, W, device=device, dtype=dtype)  # [HW,2]

        node_feats_list, adj_masks_list, labels_list, centroids_list = [], [], [], []
        HW = H * W

        for b in range(B):
            feat_map = x_c[b]  # [C,H,W]
            edge_mag = self._edge_strength_map_torch(feat_map)  # [H,W]

            pixels = feat_map.permute(1, 2, 0).reshape(HW, C)   # [HW,C]
            feat = torch.cat([pixels, coords], dim=1)           # [HW,C+2]

            # number of nodes (same logic as之前)
            n_nodes = int(round(float(HW) / max(1.0, float(self.nodes_per_pix))))
            n_nodes = max(self.min_nodes, min(self.max_nodes, n_nodes))

            labels = self._kmeans_torch(feat, n_nodes)          # [HW]
            labels_hw = labels.view(H, W)                       # [H,W]

            # node feats = mean of pixels (only C dims, not coords)
            sums = torch.zeros((n_nodes, C), device=device, dtype=dtype)
            sums.index_add_(0, labels, pixels)
            counts = torch.bincount(labels, minlength=n_nodes).clamp_min(1).to(dtype)  # [K]
            node_feats = sums / counts[:, None]                 # [K,C]

            # Geometry-defined node positions: mean normalized (x, y) coordinate
            # of all pixels assigned to each node. These centroids replace the
            # arbitrary K-Means label-index difference in spatial attention.
            coord_sums = torch.zeros((n_nodes, 2), device=device, dtype=dtype)
            coord_sums.index_add_(0, labels, coords)
            node_centroids = coord_sums / counts[:, None]       # [K,2]

            adj_mask = self._build_adj_mask(labels_hw, edge_mag, n_nodes)  # [K,K]

            node_feats_list.append(node_feats)
            adj_masks_list.append(adj_mask)
            labels_list.append(labels_hw)
            centroids_list.append(node_centroids)

        return node_feats_list, adj_masks_list, labels_list, centroids_list

    def graph_attention(
        self,
        node_feats: torch.Tensor,
        adj_mask: torch.Tensor,
        node_centroids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjacency-constrained tri-source attention:
          att = softmax( (s_sem + s_ctx + lambda*s_sp) masked )
        node_feats: [N, C]
        adj_mask  : [N, N] (0/1)
        node_centroids: [N, 2], normalized (x, y) coordinates
        return att: [N, N]
        """
        N, C = node_feats.shape
        q = self.q_proj(node_feats)  # [N,C]
        k = self.k_proj(node_feats)  # [N,C]
        s_sem = (q @ k.t()) / math.sqrt(max(1.0, float(C)))  # [N,N]

        # Context score is assigned to each candidate destination node j and
        # broadcast over source nodes i. Therefore S_ctx(i,j)=g_j varies along
        # every softmax row and can affect neighbor selection.
        z_ctx = self.ctx_proj(node_feats)                              # [N,C]
        ctx = z_ctx.mean(dim=0, keepdim=True)                          # [1,C]
        target_ctx_score = (z_ctx @ ctx.t()) / math.sqrt(max(1.0, float(C)))  # [N,1]
        s_ctx = target_ctx_score.t().expand(N, -1)                     # [N,N]

        # Spatial score uses actual node-centroid geometry rather than the
        # arbitrary numeric difference between K-Means cluster labels.
        dist = torch.cdist(node_centroids, node_centroids, p=2)        # [N,N]
        positive_dist = dist[dist > 0]
        if positive_dist.numel() > 0:
            sigma = positive_dist.median().clamp_min(1e-6)
        else:
            sigma = dist.new_tensor(1.0)
        s_sp = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))

        scores = s_sem + s_ctx + self.lambda_spatial * s_sp
        scores = scores.masked_fill(adj_mask <= 0, float("-inf"))
        att = torch.softmax(scores, dim=1)
        return att

    def consensus_propagation(
        self,
        node_feats: torch.Tensor,
        adj_mask: torch.Tensor,
        node_centroids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Iterative dense message passing:
          for t in 1..T:
             att = graph_attention(h, adj_mask)
             h = LayerNorm( h + f(att@h) )
        """
        h = node_feats
        for _ in range(self.T):
            att = self.graph_attention(h, adj_mask, node_centroids)
            msg = att @ h
            for lin in self.gnn_layers:
                msg = lin(msg)
                msg = F.relu(msg, inplace=True)
                if self.dropout > 0:
                    msg = F.dropout(msg, p=self.dropout, training=self.training)
            h = self.norm(h + msg)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        return: [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        x_c = self.channel_compress(x)  # [B, feat_dim, H, W]

        # ---- graph branch (GPU KMeans + GPU Sobel) ----
        # 注意：建图/聚类本身不可导，我们显式 no_grad，避免显存膨胀
        with torch.no_grad():
            node_feats_list, adj_masks_list, labels_list, centroids_list = self.build_spatial_nodes(x_c)

        graph_maps = []
        for b in range(B):
            h_T = self.consensus_propagation(
                node_feats_list[b], adj_masks_list[b], centroids_list[b]
            )                                                                        # [K,feat_dim]
            node_out = self.spatial_proj(h_T)                                       # [K,out]
            labels_hw = labels_list[b]                                              # [H,W]
            pix = node_out[labels_hw.reshape(-1)].view(H, W, self.out_channels)     # [H,W,out]
            graph_maps.append(pix.permute(2, 0, 1).contiguous())
        graph_feature_maps = torch.stack(graph_maps, dim=0)  # [B,out,H,W]

        # ---- CNN branch ----
        cnn_feature_maps = self.local_cnn_branch(x_c)

        # ---- fuse ----
        alpha = torch.clamp(self._alpha_buf, 0.0, 1.0)
        out = (1.0 - alpha) * graph_feature_maps + alpha * cnn_feature_maps
        return out


class GraphCrossAttention(nn.Module):
    """Graph Cross Attention (GCA): cross-attention between graph-enhanced features and morphological priors."""
    def __init__(self, channels: int, heads: int = 8, morph_ratio: float = 0.25,
                 dropout: float = 0.0, residual_scale: float = 0.5):
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        assert self.channels % self.heads == 0, f"channels={self.channels} 必须能被 heads={self.heads} 整除"
        morph_ch = max(8, int(round(self.channels * float(morph_ratio))))
        self.residual_scale = float(residual_scale)

        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, morph_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(morph_ch),
            nn.ReLU(inplace=True),
        )
        self.struct_proj = nn.Sequential(
            nn.Conv2d(1, morph_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(morph_ch),
            nn.ReLU(inplace=True),
        )
        inter_in = self.channels + morph_ch * 2
        self.interaction = nn.Sequential(
            nn.Conv2d(inter_in, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.q_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.k_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.v_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.q_norm = nn.LayerNorm(self.channels)
        self.k_norm = nn.LayerNorm(self.channels)
        self.v_norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(self.channels, self.heads, dropout=float(dropout), batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, b_g: torch.Tensor, e_morph: torch.Tensor, s_morph: torch.Tensor) -> torch.Tensor:
        B, C, H, W = b_g.shape
        if e_morph.shape[-2:] != (H, W):
            e_morph = F.interpolate(e_morph, size=(H, W), mode="bilinear", align_corners=False)
        if s_morph.shape[-2:] != (H, W):
            s_morph = F.interpolate(s_morph, size=(H, W), mode="bilinear", align_corners=False)

        f_e = self.edge_proj(e_morph)
        f_s = self.struct_proj(s_morph)
        f_int = self.interaction(torch.cat([b_g, f_e, f_s], dim=1))

        q = self.q_proj(b_g).flatten(2).transpose(1, 2)
        k = self.k_proj(f_int).flatten(2).transpose(1, 2)
        v = self.v_proj(f_int).flatten(2).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        delta, _ = self.attn(q, k, v, need_weights=False)
        delta = delta.transpose(1, 2).reshape(B, C, H, W)
        delta = self.out_proj(delta)
        return b_g + self.residual_scale * delta
