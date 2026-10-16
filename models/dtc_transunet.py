"""DTC-TransUNet architecture."""
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (
    ConvBNReLU, DoubleConv, Down, Up,
    MorphologicalPriorExtractor, DynamicGatingModulation,
    BottleneckTransformer, DynamicTriSourceConsensusGCN, GraphCrossAttention,
)

# -----------------------------
# DTC-TransUNet main architecture
# -----------------------------

class DTCTransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        base: int = 64,
        trans_depth: int = 4,
        trans_heads: int = 8,
        dtc_gcn_layers: int = 2,
        dtc_gcn_dropout: float = 0.0,
        dtc_gcn_T: int = 3,
        dtc_gcn_alpha_fuse: float = 0.5,
        dtc_gcn_nodes_per_pix: int = 512,
        dtc_gcn_max_nodes: int = 400,
        use_morph: bool = True,
        morph_mid: int = 32,
        morph_alpha: float = 0.7,
        morph_beta: float = 0.5,
        use_gca: bool = True,
        gca_heads: int = 8,
        gca_dropout: float = 0.0,
        gca_residual_scale: float = 0.5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base = int(base)

        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.use_morph = bool(use_morph)
        self.morph_alpha = float(morph_alpha)
        self.morph_beta = float(morph_beta)
        if self.use_morph:
            self.morph = MorphologicalPriorExtractor(
                in_img=in_channels,
                c2=base * 4,
                c3=base * 8,
                mid_channels=int(morph_mid),
            )
            self.gate_s1 = DynamicGatingModulation(base)
            self.gate_s2 = DynamicGatingModulation(base * 2)
            self.gate_s3 = DynamicGatingModulation(base * 4)
            self.gate_b  = DynamicGatingModulation(base * 8)
        else:
            self.morph = None


        self.dtc_gcn = DynamicTriSourceConsensusGCN(
            in_channels=base * 8,
            out_channels=base * 8,
            gnn_layers=int(dtc_gcn_layers),
            T=int(dtc_gcn_T),
            alpha_fuse=float(dtc_gcn_alpha_fuse),
            nodes_per_pix=int(dtc_gcn_nodes_per_pix),
            max_nodes=int(dtc_gcn_max_nodes),
            dropout=float(dtc_gcn_dropout),
        )

        self.use_gca = bool(use_gca) and self.use_morph
        if self.use_gca:
            self.gca = GraphCrossAttention(
                channels=base * 8,
                heads=int(gca_heads),
                dropout=float(gca_dropout),
                residual_scale=float(gca_residual_scale),
            )
        else:
            self.gca = None

        self.bottleneck_trans = BottleneckTransformer(dim=base * 8, depth=trans_depth, nhead=trans_heads)

        self.up1 = Up(base * 8, skip_ch=base * 4, out_ch=base * 4)
        self.up2 = Up(base * 4, skip_ch=base * 2, out_ch=base * 2)
        self.up3 = Up(base * 2, skip_ch=base,     out_ch=base)
        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, x, return_aux: bool = False):
        aux = {}
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b  = self.down3(s3)

        if self.use_morph:
            e_morph, s_morph = self.morph(x, s3, b)
            aux["e_morph"] = e_morph
            aux["s_morph"] = s_morph
            s1 = self.gate_s1(s1, e_morph, s_morph, alpha=self.morph_alpha, beta=self.morph_beta)
            s2 = self.gate_s2(s2, e_morph, s_morph, alpha=self.morph_alpha, beta=self.morph_beta)
            s3 = self.gate_s3(s3, e_morph, s_morph, alpha=self.morph_alpha, beta=self.morph_beta)
            b  = self.gate_b (b,  e_morph, s_morph, alpha=self.morph_alpha, beta=self.morph_beta)


        b_g = self.dtc_gcn(b)
        aux["f_dtc"] = b_g

        if self.use_gca and ("e_morph" in aux) and ("s_morph" in aux):
            b_g = self.gca(b_g, aux["e_morph"], aux["s_morph"])
            aux["f_gca"] = b_g

        b_prime = self.bottleneck_trans(b_g)


        x = self.up1(b_prime, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        logits = self.outc(x)

        if return_aux:
            aux["logits"] = logits
            return aux
        return logits


# -----------------------------
# 7) Xavier init (paper-friendly)
# -----------------------------


def apply_xavier_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            if getattr(m, "weight", None) is not None and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
