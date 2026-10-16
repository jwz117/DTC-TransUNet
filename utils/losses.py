"""Segmentation losses and auxiliary morphological supervision."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 8) Loss & metrics
# -----------------------------

def dice_loss(prob1: torch.Tensor, target01: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss on foreground probability."""
    prob1 = prob1.clamp(0.0, 1.0)
    target01 = target01.clamp(0.0, 1.0)
    inter = (prob1 * target01).sum(dim=(1, 2, 3))
    denom = prob1.pow(2).sum(dim=(1, 2, 3)) + target01.pow(2).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha: float = 0.7, beta: float = 0.3,
                 smooth: float = 1.0) -> torch.Tensor:
    """Tversky loss：对 FP / FN 不平衡敏感（适合前景稀疏的二分类分割）。"""
    prob = prob.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky).mean()


class CombinedLoss(nn.Module):
    """
    CE（带 label smoothing） + Dice + Tversky 的加权组合。
    """
    def __init__(self,
                 w_ce: float = 0.3,
                 w_dice: float = 0.4,
                 w_tversky: float = 0.3,
                 tversky_alpha: float = 0.7,
                 tversky_beta: float = 0.3,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        self.w_ce = float(w_ce)
        self.w_dice = float(w_dice)
        self.w_tversky = float(w_tversky)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)

    def forward(self, logits: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        # logits: [B,2,H,W], target01: [B,1,H,W]
        target_cls = target01.squeeze(1).long()
        l_ce = self.ce(logits, target_cls)

        prob1 = torch.softmax(logits, dim=1)[:, 1:2]
        l_dice = dice_loss(prob1, target01)
        l_tversky = tversky_loss(prob1, target01, alpha=self.tversky_alpha, beta=self.tversky_beta)

        return self.w_ce * l_ce + self.w_dice * l_dice + self.w_tversky * l_tversky





def soft_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    target = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred, target)


def make_boundary_target(mask01: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dil = F.max_pool2d(mask01, kernel_size=kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=kernel_size, stride=1, padding=pad)
    edge = (dil - ero).clamp(0.0, 1.0)
    return (edge > 0).float()


def make_structure_target(mask01: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    pad = kernel_size // 2
    soft_region = F.avg_pool2d(mask01, kernel_size=kernel_size, stride=1, padding=pad)
    soft_region = (soft_region * mask01).clamp(0.0, 1.0)
    return soft_region


def morph_aux_loss(e_pred: torch.Tensor, s_pred: torch.Tensor, target01: torch.Tensor,
                   edge_w: float = 0.2, struct_w: float = 0.2):
    edge_target = make_boundary_target(target01)
    struct_target = make_structure_target(target01)

    l_edge_bce = soft_bce_loss(e_pred, edge_target)
    l_edge_dice = dice_loss(e_pred, edge_target)
    l_edge = 0.7 * l_edge_bce + 0.3 * l_edge_dice

    l_struct_bce = soft_bce_loss(s_pred, struct_target)
    l_struct_l1 = F.l1_loss(s_pred, struct_target)
    l_struct = 0.7 * l_struct_bce + 0.3 * l_struct_l1

    total = float(edge_w) * l_edge + float(struct_w) * l_struct
    return total, {
        "edge_total": l_edge.detach(),
        "struct_total": l_struct.detach(),
        "edge_target": edge_target.detach(),
        "struct_target": struct_target.detach(),
    }
