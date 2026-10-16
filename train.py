"""Command-line entry point for training and evaluation."""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.npy_dataset import NpyLandslideDataset, create_loaders
from models.dtc_transunet import DTCTransUNet
from trainer import Trainer, TrainConfig, _resolve_device
from utils.checkpoint import load_model_checkpoint
from utils.reproducibility import set_seed

RED = "\033[31m"
WHITE = "\033[37m"
RESET = "\033[0m"

# -----------------------------
# 10) CLI
# -----------------------------


def _load_npy(paths):
    for p in paths:
        if p is None:
            continue
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 NPY 文件: {p}")
    arrs = [np.load(p) if p is not None else None for p in paths]
    for p, a in zip(paths, arrs):
        if p is None:
            continue
        print(f"{os.path.basename(p)} shape={a.shape}, dtype={a.dtype}")
    return arrs


def build_parser():
    p = argparse.ArgumentParser("DTC-TransUNet for binary landslide segmentation")
    sub = p.add_subparsers(dest="mode", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--gpu_id", type=int, default=0, help="GPU index")
    tr.add_argument("--allow_cpu", action="store_true", help="allow CPU execution when CUDA is unavailable")
    tr.add_argument("--train_images", type=str, default=r"data/train_images.npy")
    tr.add_argument("--train_labels", type=str, default=r"data/train_labels.npy")
    tr.add_argument("--val_images", type=str, default=r"data/val_images.npy")
    tr.add_argument("--val_labels", type=str, default=r"data/val_labels.npy")
    tr.add_argument("--test_images", type=str, default=r"data/test_images.npy")
    tr.add_argument("--test_labels", type=str, default=r"data/test_labels.npy")
    tr.add_argument("--skip_final_test", action="store_true", help="训练结束后不在独立测试集上评估 best_model")

    tr.add_argument("--batch_size", type=int, default=16)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    tr.add_argument("--num_workers", type=int, default=0)

    # aug
    tr.add_argument("--use_aug", action="store_true")
    tr.add_argument("--aug_prob", type=float, default=0.7)
    tr.add_argument("--brightness", type=float, default=0.15)
    tr.add_argument("--contrast", type=float, default=0.15)
    tr.add_argument("--noise_std", type=float, default=0.02)
    tr.add_argument("--no_rot90", action="store_true")
    tr.add_argument("--no_hflip", action="store_true")
    tr.add_argument("--no_vflip", action="store_true")

    # model
    tr.add_argument("--base", type=int, default=64)
    tr.add_argument("--trans_depth", type=int, default=4)
    tr.add_argument("--trans_heads", type=int, default=8)
    tr.add_argument("--dtc_gcn_layers", type=int, default=2)
    tr.add_argument("--dtc_gcn_dropout", type=float, default=0.0)
    # DTC-GCN hyperparameters
    tr.add_argument("--dtc_gcn_T", type=int, default=3)
    tr.add_argument("--dtc_gcn_alpha_fuse", type=float, default=0.5)
    tr.add_argument("--dtc_gcn_nodes_per_pix", type=int, default=512)
    tr.add_argument("--dtc_gcn_max_nodes", type=int, default=400)

    # morph
    tr.add_argument("--no_morph", action="store_true", help="disable MPE and DGM (ablation only)")
    tr.add_argument("--morph_mid", type=int, default=32)
    tr.add_argument("--morph_alpha", type=float, default=0.7)
    tr.add_argument("--morph_beta", type=float, default=0.5)
    tr.add_argument("--no_gca", action="store_true", help="disable Graph Cross Attention (GCA)")
    tr.add_argument("--gca_heads", type=int, default=8)
    tr.add_argument("--gca_dropout", type=float, default=0.0)
    tr.add_argument("--gca_residual_scale", type=float, default=0.5)


    # optim
    tr.add_argument("--base_lr", type=float, default=1e-2)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--momentum", type=float, default=0.9)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--edge_loss_w", type=float, default=0.2)
    tr.add_argument("--struct_loss_w", type=float, default=0.2)

    tr.add_argument("--seed", type=int, default=1234)
    tr.add_argument("--deterministic", type=int, default=1)
    tr.add_argument("--n_gpu", type=int, default=1)
    tr.add_argument("--visualize_freq", type=int, default=5)
    tr.add_argument("--result_dir", type=str, default=None)
    tr.add_argument("--no_xavier", action="store_true")

    # test
    te = sub.add_parser("test")
    te.add_argument("--gpu_id", type=int, default=0, help="GPU index")
    te.add_argument("--allow_cpu", action="store_true", help="allow CPU execution when CUDA is unavailable")
    te.add_argument("--test_images", "--val_images", dest="test_images", type=str, default=r"data/test_images.npy")
    te.add_argument("--test_labels", "--val_labels", dest="test_labels", type=str, default=r"data/test_labels.npy")
    te.add_argument("--ckpt_path", type=str, required=True)
    te.add_argument("--norm_stats", type=str, default=None, help="训练时保存的 norm_stats.npz；默认读取 ckpt 同目录下的 norm_stats.npz")
    te.add_argument("--batch_size", type=int, default=16)
    te.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    te.add_argument("--num_workers", type=int, default=0)

    te.add_argument("--base", type=int, default=64)
    te.add_argument("--trans_depth", type=int, default=4)
    te.add_argument("--trans_heads", type=int, default=8)
    te.add_argument("--dtc_gcn_layers", type=int, default=2)
    te.add_argument("--dtc_gcn_dropout", type=float, default=0.0)
    # DTC-GCN hyper-params
    te.add_argument("--dtc_gcn_T", type=int, default=3)
    te.add_argument("--dtc_gcn_alpha_fuse", type=float, default=0.5)
    te.add_argument("--dtc_gcn_nodes_per_pix", type=int, default=512)
    te.add_argument("--dtc_gcn_max_nodes", type=int, default=400)

    te.add_argument("--no_morph", action="store_true")
    te.add_argument("--morph_mid", type=int, default=32)
    te.add_argument("--morph_alpha", type=float, default=0.7)
    te.add_argument("--morph_beta", type=float, default=0.5)
    te.add_argument("--no_gca", action="store_true")
    te.add_argument("--gca_heads", type=int, default=8)
    te.add_argument("--gca_dropout", type=float, default=0.0)
    te.add_argument("--gca_residual_scale", type=float, default=0.5)


    te.add_argument("--seed", type=int, default=1234)
    te.add_argument("--deterministic", type=int, default=1)
    te.add_argument("--n_gpu", type=int, default=1)
    te.add_argument("--no_xavier", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    if args.mode is None:
        raise ValueError("Please specify a mode: train or test.")

    # device
    device = _resolve_device(allow_cpu=bool(args.allow_cpu), gpu_id=int(args.gpu_id))

    set_seed(args.seed, deterministic=bool(args.deterministic))
    img_h, img_w = tuple(args.img_size)

    if args.mode == "train":
        train_images, train_labels, val_images, val_labels = _load_npy(
            [args.train_images, args.train_labels, args.val_images, args.val_labels]
        )
        if not bool(args.skip_final_test):
            test_images, test_labels = _load_npy([args.test_images, args.test_labels])
        else:
            test_images, test_labels = None, None

        if train_images.ndim != 4:
            raise ValueError(f"train_images 应为 [N,C,H,W]，当前 {train_images.shape}")
        if train_labels.ndim not in (3, 4):
            raise ValueError(f"train_labels 应为 [N,H,W] 或 [N,1,H,W]，当前 {train_labels.shape}")

        in_channels = int(train_images.shape[1])
        print(f"{WHITE}NPY in_channels={in_channels}, img_size={args.img_size}, classes=2{RESET}")

        train_loader, val_loader = create_loaders(
            train_images,
            train_labels,
            val_images,
            val_labels,
            batch_size=int(args.batch_size),
            img_size=(img_h, img_w),
            num_workers=int(args.num_workers),
            use_aug=bool(args.use_aug),
            aug_prob=float(args.aug_prob),
            brightness=float(args.brightness),
            contrast=float(args.contrast),
            noise_std=float(args.noise_std),
            rot90=(not bool(args.no_rot90)),
            hflip=(not bool(args.no_hflip)),
            vflip=(not bool(args.no_vflip)),
        )

        test_loader = None
        if not bool(args.skip_final_test):
            test_ds = NpyLandslideDataset(
                test_images,
                test_labels,
                img_size=(img_h, img_w),
                is_train=False,
                mean=train_loader.dataset.mean,
                std=train_loader.dataset.std,
                use_aug=False,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=True,
                drop_last=False,
            )

        model = DTCTransUNet(
            in_channels=in_channels,
            num_classes=2,
            base=int(args.base),
            trans_depth=int(args.trans_depth),
            trans_heads=int(args.trans_heads),
            dtc_gcn_layers=int(args.dtc_gcn_layers),
            dtc_gcn_dropout=float(args.dtc_gcn_dropout),
            dtc_gcn_T=int(args.dtc_gcn_T),
            dtc_gcn_alpha_fuse=float(args.dtc_gcn_alpha_fuse),
            dtc_gcn_nodes_per_pix=int(args.dtc_gcn_nodes_per_pix),
            dtc_gcn_max_nodes=int(args.dtc_gcn_max_nodes),
            use_morph=(not bool(args.no_morph)),
            morph_mid=int(args.morph_mid),
            morph_alpha=float(args.morph_alpha),
            morph_beta=float(args.morph_beta),
            use_gca=(not bool(args.no_gca)),
            gca_heads=int(args.gca_heads),
            gca_dropout=float(args.gca_dropout),
            gca_residual_scale=float(args.gca_residual_scale),
        )

        result_dir = args.result_dir
        if result_dir is None:
            result_dir = os.path.join(os.getcwd(), "results", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(result_dir, exist_ok=True)
        np.savez(os.path.join(result_dir, "norm_stats.npz"), mean=train_loader.dataset.mean, std=train_loader.dataset.std)
        print(f"{WHITE}Saved normalization stats: {os.path.join(result_dir, 'norm_stats.npz')}{RESET}")

        cfg = TrainConfig(
            base_lr=float(args.base_lr),
            gamma=float(args.gamma),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            edge_loss_w=float(args.edge_loss_w),
            struct_loss_w=float(args.struct_loss_w),
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            result_dir=result_dir,
            cfg=cfg,
            device=device,
            n_gpu=int(args.n_gpu),
            visualize_freq=int(args.visualize_freq),
            use_xavier=(not bool(args.no_xavier)),
        )

        try:
            trainer.fit(epochs=int(args.epochs))
            trainer.plot_curves()

            if test_loader is not None:
                best_path = os.path.join(result_dir, "best_model.pth")
                if os.path.isfile(best_path):
                    target_model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
                    load_model_checkpoint(target_model, best_path, map_location="cpu")
                    test_miou, test_pre, test_rec, test_f1, test_acc = trainer.validate(loader=test_loader, desc="Test")
                    metrics_line = (
                        f"Test: mIoU={test_miou:.4f}, F1={test_f1:.4f}, "
                        f"P={test_pre:.4f}, R={test_rec:.4f}, OA={test_acc:.4f}"
                    )
                    print(f"{WHITE}{metrics_line}{RESET}")
                    with open(os.path.join(result_dir, "test_metrics.csv"), "w", encoding="utf-8") as f:
                        f.write("mIoU,F1,Precision,Recall,OA\n")
                        f.write(f"{test_miou:.6f},{test_f1:.6f},{test_pre:.6f},{test_rec:.6f},{test_acc:.6f}\n")
                else:
                    print(f"{RED}未找到 best_model.pth，跳过最终测试。{RESET}")

            print(f"{WHITE}Done. Results saved in: {result_dir}{RESET}")
        except Exception as e:
            print(f"{RED}训练过程中发生错误: {e}{RESET}")
            import traceback

            traceback.print_exc()
            trainer.save_ckpt("crash_model.pth")
        return

    if args.mode == "test":
        test_images, test_labels = _load_npy([args.test_images, args.test_labels])

        if test_images.ndim != 4:
            raise ValueError(f"test_images 应为 [N,C,H,W]，当前 {test_images.shape}")
        if test_labels.ndim not in (3, 4):
            raise ValueError(f"test_labels 应为 [N,H,W] 或 [N,1,H,W]，当前 {test_labels.shape}")

        in_channels = int(test_images.shape[1])
        norm_stats_path = args.norm_stats
        if norm_stats_path is None:
            norm_stats_path = os.path.join(os.path.dirname(args.ckpt_path), "norm_stats.npz")

        mean, std = None, None
        if os.path.isfile(norm_stats_path):
            norm_stats = np.load(norm_stats_path)
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            print(f"{WHITE}Loaded normalization stats: {norm_stats_path}{RESET}")
        else:
            print(f"{RED}警告：未找到训练集归一化参数 {norm_stats_path}，将临时使用测试集自身统计量，不建议用于论文最终测试。{RESET}")

        val_ds = NpyLandslideDataset(
            test_images, test_labels, img_size=(img_h, img_w), is_train=False, mean=mean, std=std, use_aug=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

        model = DTCTransUNet(
            in_channels=in_channels,
            num_classes=2,
            base=int(args.base),
            trans_depth=int(args.trans_depth),
            trans_heads=int(args.trans_heads),
            dtc_gcn_layers=int(args.dtc_gcn_layers),
            dtc_gcn_dropout=float(args.dtc_gcn_dropout),
            dtc_gcn_T=int(args.dtc_gcn_T),
            dtc_gcn_alpha_fuse=float(args.dtc_gcn_alpha_fuse),
            dtc_gcn_nodes_per_pix=int(args.dtc_gcn_nodes_per_pix),
            dtc_gcn_max_nodes=int(args.dtc_gcn_max_nodes),
            use_morph=(not bool(args.no_morph)),
            morph_mid=int(args.morph_mid),
            morph_alpha=float(args.morph_alpha),
            morph_beta=float(args.morph_beta),
            use_gca=(not bool(args.no_gca)),
            gca_heads=int(args.gca_heads),
            gca_dropout=float(args.gca_dropout),
            gca_residual_scale=float(args.gca_residual_scale),
        )

        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {args.ckpt_path}")
        load_model_checkpoint(model, args.ckpt_path, map_location="cpu")

        # evaluator reuse validate
        cfg = TrainConfig()
        evaluator = Trainer(
            model=model,
            train_loader=val_loader,
            val_loader=val_loader,
            result_dir=os.path.dirname(args.ckpt_path),
            cfg=cfg,
            device=device,
            n_gpu=int(args.n_gpu),
            visualize_freq=999999,
            use_xavier=False,
        )

        miou, pre, rec, f1, acc = evaluator.validate(desc="Test")
        print(f"{WHITE}Test: mIoU={miou:.4f}, F1={f1:.4f}, P={pre:.4f}, R={rec:.4f}, OA={acc:.4f}{RESET}")
        return


if __name__ == "__main__":
    main()
