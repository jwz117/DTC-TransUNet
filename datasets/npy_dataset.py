"""NPY dataset and DataLoader utilities."""
import random
from typing import Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Dataset (NPY)
# -----------------------------

class NpyLandslideDataset(Dataset):
    """images: [N,C,H,W] float32
       masks : [N,1,H,W] or [N,H,W] float32 in {0,1}
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        img_size: Tuple[int, int] = (128, 128),
        is_train: bool = True,
        mean=None,
        std=None,
        use_aug: bool = False,
        aug_prob: float = 0.7,
        brightness: float = 0.15,
        contrast: float = 0.15,
        noise_std: float = 0.02,
        rot90: bool = True,
        hflip: bool = True,
        vflip: bool = True,
    ):
        super().__init__()
        assert images.ndim == 4, f"images 应为 [N,C,H,W]，当前 {images.shape}"
        assert masks.ndim in (3, 4), f"masks 应为 [N,H,W] 或 [N,1,H,W]，当前 {masks.shape}"
        assert images.shape[0] == masks.shape[0], "images/masks 样本数不一致"

        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.img_size = tuple(img_size)
        self.is_train = bool(is_train)

        if self.masks.ndim == 3:
            self.masks = self.masks[:, None, :, :]
        if self.masks.shape[1] != 1:
            raise ValueError(f"masks 第二维应为 1，当前 {self.masks.shape}")

        self.N, self.C, _, _ = self.images.shape

        if mean is None or std is None:
            self.mean = self.images.mean(axis=(0, 2, 3))
            self.std = self.images.std(axis=(0, 2, 3)) + 1e-6
        else:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32) + 1e-6

        self.use_aug = bool(use_aug) and self.is_train
        self.aug_prob = float(aug_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.noise_std = float(noise_std)
        self.rot90 = bool(rot90)
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)

        print(f"NPY Dataset: N={self.N}, C={self.C}, is_train={self.is_train}, use_aug={self.use_aug}")
        print(f"mean={self.mean}")
        print(f"std ={self.std}")

    def __len__(self):
        return self.N

    @staticmethod
    def _apply_flip_rot(img, mask, hflip=True, vflip=True, rot90=True):
        if hflip and np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        if vflip and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        if rot90:
            k = np.random.randint(0, 4)
            if k != 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return img, mask

    def _apply_brightness_contrast(self, img):
        ch = 3 if self.C >= 3 else self.C
        b = (np.random.rand() * 2 - 1) * self.brightness if self.brightness > 0 else 0.0
        c = 1.0 + (np.random.rand() * 2 - 1) * self.contrast if self.contrast > 0 else 1.0
        out = img.copy()
        out[:ch] = out[:ch] * c + b
        return out

    def _apply_noise(self, img):
        if self.noise_std <= 0:
            return img
        noise = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
        return img + noise

    def __getitem__(self, idx):
        img = self.images[idx]  # [C,H,W]
        mask = self.masks[idx]  # [1,H,W]

        th, tw = self.img_size
        if (img.shape[1], img.shape[2]) != (th, tw):
            resized = np.zeros((self.C, th, tw), dtype=np.float32)
            for c in range(self.C):
                resized[c] = cv2.resize(img[c], (tw, th), interpolation=cv2.INTER_LINEAR)
            img = resized
            m0 = cv2.resize(mask[0], (tw, th), interpolation=cv2.INTER_NEAREST)
            mask = m0[None, :, :]

        if self.use_aug and (np.random.rand() < self.aug_prob):
            img, mask = self._apply_flip_rot(img, mask, self.hflip, self.vflip, self.rot90)
            img = self._apply_brightness_contrast(img)
            img = self._apply_noise(img)

        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        img = np.ascontiguousarray(img, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)

        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask).float().clamp(0.0, 1.0)
        return img_t, mask_t


def create_loaders(
    train_images,
    train_labels,
    val_images,
    val_labels,
    batch_size=16,
    img_size=(128, 128),
    num_workers=0,
    use_aug=False,
    aug_prob=0.7,
    brightness=0.15,
    contrast=0.15,
    noise_std=0.02,
    rot90=True,
    hflip=True,
    vflip=True,
):
    train_ds = NpyLandslideDataset(
        train_images,
        train_labels,
        img_size=img_size,
        is_train=True,
        mean=None,
        std=None,
        use_aug=use_aug,
        aug_prob=aug_prob,
        brightness=brightness,
        contrast=contrast,
        noise_std=noise_std,
        rot90=rot90,
        hflip=hflip,
        vflip=vflip,
    )
    val_ds = NpyLandslideDataset(
        val_images,
        val_labels,
        img_size=img_size,
        is_train=False,
        mean=train_ds.mean,
        std=train_ds.std,
        use_aug=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
