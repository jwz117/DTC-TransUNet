"""Segmentation evaluation metrics."""
import torch

@torch.no_grad()
def binary_miou_from_confusion(tn: int, fp: int, fn: int, tp: int, eps: float = 1e-7) -> float:
    """
    标准二分类 mIoU：背景类 IoU 与滑坡前景类 IoU 的算术平均。
    tn/fp/fn/tp 应为整个评估集累计的像素计数。
    """
    iou_bg = tn / (tn + fp + fn + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    return float((iou_bg + iou_fg) / 2.0)
