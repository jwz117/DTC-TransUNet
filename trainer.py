"""Training loop, validation, checkpoints, and visualizations."""
import os
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.dtc_transunet import apply_xavier_init
from utils.losses import CombinedLoss, morph_aux_loss
from utils.metrics import binary_miou_from_confusion

RED = "\033[31m"
WHITE = "\033[37m"
RESET = "\033[0m"

# -----------------------------
# 9) Trainer
# -----------------------------


@dataclass
class TrainConfig:
    base_lr: float = 1e-2
    gamma: float = 0.99
    momentum: float = 0.9
    weight_decay: float = 1e-4
    w_ce: float = 0.5
    w_dice: float = 0.5
    edge_loss_w: float = 0.2
    struct_loss_w: float = 0.2


def _resolve_device(allow_cpu: bool, gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(int(gpu_id))
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(dev)
        print(f"{WHITE}[DEVICE] Using CUDA: {name}{RESET}")
        return dev

    if allow_cpu:
        print(f"{RED}[DEVICE] CUDA 不可用，已退回 CPU（你使用了 --allow_cpu）。{RESET}")
        return torch.device("cpu")

    raise RuntimeError("未检测到可用的 CUDA/GPU。若你确定要在 CPU 上运行，请加参数 --allow_cpu。")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        result_dir: str,
        cfg: TrainConfig,
        device: torch.device,
        n_gpu: int = 1,
        visualize_freq: int = 5,
        use_xavier: bool = True,
    ):
        self.device = device

        if use_xavier:
            apply_xavier_init(model)

        if self.device.type == "cuda" and n_gpu > 1 and torch.cuda.device_count() >= n_gpu:
            model = nn.DataParallel(model, device_ids=list(range(n_gpu)))

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.cfg = cfg
        self.visualize_freq = int(visualize_freq)

        self.criterion = CombinedLoss(
            w_ce=0.3, w_dice=0.4, w_tversky=0.3,
            tversky_alpha=0.65, tversky_beta=0.35
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.gamma)

        self.best_miou = -1.0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_seg_loss": [],
            "train_edge_loss": [],
            "train_struct_loss": [],
            "val_miou": [],
            "val_f1": [],
            "val_acc": [],
            "val_recall": [],
            "lr": [],
        }

        pdev = next(self.model.parameters()).device
        print(f"{WHITE}[SANITY] model param device = {pdev}{RESET}")

    def _forward_loss(self, images, masks01):
        out = self.model(images, return_aux=True)
        logits = out["logits"]
        seg_loss = self.criterion(logits, masks01)
        total_loss = seg_loss

        edge_loss = torch.tensor(0.0, device=images.device)
        struct_loss = torch.tensor(0.0, device=images.device)
        if ("e_morph" in out) and ("s_morph" in out):
            aux_loss, aux_info = morph_aux_loss(
                out["e_morph"], out["s_morph"], masks01,
                edge_w=self.cfg.edge_loss_w,
                struct_w=self.cfg.struct_loss_w,
            )
            total_loss = total_loss + aux_loss
            edge_loss = aux_info["edge_total"]
            struct_loss = aux_info["struct_total"]

        prob1 = torch.softmax(logits, dim=1)[:, 1:2]
        loss_dict = {
            "total": total_loss,
            "seg": seg_loss.detach(),
            "edge": edge_loss,
            "struct": struct_loss,
        }
        return total_loss, prob1, loss_dict, out

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_seg = 0.0
        total_edge = 0.0
        total_struct = 0.0

        for images, masks in tqdm(self.train_loader, desc=f"{RED}Train {epoch+1}{RESET}", ncols=90):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss, _, loss_dict, _ = self._forward_loss(images, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_seg += float(loss_dict["seg"].item())
            total_edge += float(loss_dict["edge"].item())
            total_struct += float(loss_dict["struct"].item())

        n = max(1, len(self.train_loader))
        return {
            "total": total_loss / n,
            "seg": total_seg / n,
            "edge": total_edge / n,
            "struct": total_struct / n,
        }

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None, desc: str = "Val"):
        """
        在整个评估集上累计像素级混淆矩阵，再统一计算指标。
        mIoU = (IoU_background + IoU_landslide) / 2。
        """
        self.model.eval()
        eval_loader = self.val_loader if loader is None else loader

        tn = fp = fn = tp = 0

        for images, masks in tqdm(eval_loader, desc=f"{RED}{desc}{RESET}", ncols=90):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            pred = torch.argmax(logits, dim=1)          # [B,H,W], class 0/1
            gt = (masks.squeeze(1) > 0.5).long()       # [B,H,W]

            tn += int(((pred == 0) & (gt == 0)).sum().item())
            fp += int(((pred == 1) & (gt == 0)).sum().item())
            fn += int(((pred == 0) & (gt == 1)).sum().item())
            tp += int(((pred == 1) & (gt == 1)).sum().item())

        eps = 1e-7
        miou = binary_miou_from_confusion(tn, fp, fn, tp, eps=eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        accuracy = (tn + tp) / (tn + fp + fn + tp + eps)

        return float(miou), float(precision), float(recall), float(f1), float(accuracy)

    def save_ckpt(self, name: str):
        path = os.path.join(self.result_dir, name)
        state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save(state, path)
        print(f"{WHITE}Saved: {path}{RESET}")

    def plot_curves(self):
        fig = plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.history["train_loss"], label="total")
        plt.plot(self.history["train_seg_loss"], label="seg")
        plt.plot(self.history["train_edge_loss"], label="edge")
        plt.plot(self.history["train_struct_loss"], label="struct")
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(2, 2, 2); plt.plot(self.history["val_miou"]); plt.title("Val mIoU"); plt.xlabel("Epoch"); plt.ylabel("mIoU")
        plt.subplot(2, 2, 3); plt.plot(self.history["val_f1"]); plt.title("Val F1"); plt.xlabel("Epoch"); plt.ylabel("F1")
        plt.subplot(2, 2, 4); plt.plot(self.history["lr"]); plt.title("LR"); plt.xlabel("Epoch"); plt.ylabel("lr")
        plt.tight_layout()
        out = os.path.join(self.result_dir, "curves.png")
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"{WHITE}Saved curves: {out}{RESET}")

    @torch.no_grad()
    def visualize(self, epoch: int, num_samples: int = 4):
        self.model.eval()
        images, masks = next(iter(self.val_loader))
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        out = self.model(images, return_aux=True)
        logits = out["logits"]
        prob1 = torch.softmax(logits, dim=1)[:, 1:2]
        pred = (prob1 > 0.5).float()

        imgs_vis = images[:, :min(3, images.size(1))].detach().cpu()
        B = imgs_vis.size(0)

        disp = []
        for i in range(B):
            x = imgs_vis[i]
            mn, mx = x.min(), x.max()
            if float(mx - mn) < 1e-6:
                x = torch.zeros_like(x)
            else:
                x = (x - mn) / (mx - mn)
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            elif x.size(0) == 2:
                x = torch.cat([x, torch.zeros(1, *x.shape[1:])], dim=0)
            disp.append(x)
        disp = torch.stack(disp, 0).permute(0, 2, 3, 1).numpy()

        gt = masks.squeeze(1).detach().cpu().numpy()
        pr = pred.squeeze(1).detach().cpu().numpy()
        e_vis = out.get("e_morph", None)
        s_vis = out.get("s_morph", None)
        if e_vis is not None:
            e_vis = e_vis.squeeze(1).detach().cpu().numpy()
        if s_vis is not None:
            s_vis = s_vis.squeeze(1).detach().cpu().numpy()

        nshow = min(num_samples, B)
        cols = 5 if (e_vis is not None and s_vis is not None) else 3
        plt.figure(figsize=(5 * cols, 4 * nshow))
        for i in range(nshow):
            plt.subplot(nshow, cols, i * cols + 1); plt.imshow(disp[i]); plt.title("Input"); plt.axis("off")
            plt.subplot(nshow, cols, i * cols + 2); plt.imshow(gt[i], cmap="gray"); plt.title("GT"); plt.axis("off")
            plt.subplot(nshow, cols, i * cols + 3); plt.imshow(pr[i], cmap="gray"); plt.title("Pred"); plt.axis("off")
            if cols == 5:
                plt.subplot(nshow, cols, i * cols + 4); plt.imshow(e_vis[i], cmap="magma"); plt.title("e_morph"); plt.axis("off")
                plt.subplot(nshow, cols, i * cols + 5); plt.imshow(s_vis[i], cmap="viridis"); plt.title("s_morph"); plt.axis("off")

        out_path = os.path.join(self.result_dir, f"vis_epoch_{epoch+1}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"{WHITE}Saved visualization: {out_path}{RESET}")

    def fit(self, epochs: int = 200):
        for epoch in range(epochs):
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(lr_now)

            train_stats = self.train_one_epoch(epoch)
            val_miou, val_pre, val_rec, val_f1, val_acc = self.validate()

            self.scheduler.step()

            self.history["train_loss"].append(train_stats["total"])
            self.history["train_seg_loss"].append(train_stats["seg"])
            self.history["train_edge_loss"].append(train_stats["edge"])
            self.history["train_struct_loss"].append(train_stats["struct"])
            self.history["val_miou"].append(val_miou)
            self.history["val_f1"].append(val_f1)
            self.history["val_acc"].append(val_acc)
            self.history["val_recall"].append(val_rec)

            print(
                f"{WHITE}Epoch {epoch+1}/{epochs}{RESET} | "
                f"lr={lr_now:.4e} | train_loss={train_stats['total']:.4f} | seg={train_stats['seg']:.4f} | edge={train_stats['edge']:.4f} | struct={train_stats['struct']:.4f} | "
                f"val_mIoU={val_miou:.4f} val_F1={val_f1:.4f} "
                f"val_P={val_pre:.4f} val_R={val_rec:.4f} val_OA={val_acc:.4f}{RESET}"
            )

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_ckpt("best_model.pth")

            if (epoch + 1) == epochs or (epoch + 1) % 50 == 0:
                self.save_ckpt(f"epoch_{epoch+1}.pth")

            if (epoch + 1) % self.visualize_freq == 0 or (epoch + 1) == epochs:
                self.visualize(epoch)
                self.plot_curves()

        print(f"{WHITE}Training finished. Best val mIoU={self.best_miou:.4f}{RESET}")
