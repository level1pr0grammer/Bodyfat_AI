#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a gender classifier using a CUSTOM CNN (no pretrained / no torchvision.models).
Directory:
  data/
    train/
      female/
      male/
    [val/]   # optional; if missing, will split from train

Example:
  python train_custom_cnn.py --data_dir data --epochs 20 --img_size 224 --batch_size 32
"""

import argparse, json, os, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ------------------------- Utils -------------------------
def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir, batch_size=32, img_size=224, val_ratio=0.15, num_workers=2):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dir = Path(data_dir) / "train"
    val_dir   = Path(data_dir) / "val"

    if val_dir.exists():
        ds_train = datasets.ImageFolder(str(train_dir), transform=tfm_train)
        ds_val   = datasets.ImageFolder(str(val_dir),   transform=tfm_eval)
        class_to_idx = ds_train.class_to_idx
    else:
        ds_full = datasets.ImageFolder(str(train_dir), transform=tfm_train)
        n_total = len(ds_full)
        n_val   = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        ds_train, ds_val = random_split(ds_full, [n_train, n_val])
        # ให้ชุด val ใช้ tfm_eval
        if isinstance(ds_val, torch.utils.data.Subset):
            ds_val.dataset.transform = tfm_eval
        class_to_idx = ds_train.dataset.class_to_idx if isinstance(ds_train, torch.utils.data.Subset) else ds_train.class_to_idx

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return loader_train, loader_val, class_to_idx


# ------------------------- Custom CNN -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """ MobileNet-style block (depthwise + pointwise) """
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=s, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class CustomCNN(nn.Module):
    """
    Lightweight CNN from scratch with:
      - initial stem
      - depthwise separable stacks
      - adaptive avg pool to (1,1) so it works with any img_size
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, k=3, s=2, p=1),    # /2
            ConvBNAct(32, 32, k=3, s=1, p=1),
        )
        # Stacks
        self.block1 = nn.Sequential(
            DepthwiseSeparable(32, 64, s=2),    # /2
            DepthwiseSeparable(64, 64, s=1),
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparable(64, 128, s=2),   # /2
            DepthwiseSeparable(128, 128, s=1),
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparable(128, 256, s=2),  # /2
            DepthwiseSeparable(256, 256, s=1),
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparable(256, 512, s=2),  # /2
            DepthwiseSeparable(512, 512, s=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)          # (N, 512, 1, 1)
        x = torch.flatten(x, 1)   # (N, 512)
        x = self.dropout(x)
        x = self.fc(x)            # logits
        return x


# ------------------------- Train / Eval -------------------------
@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    loader_train, loader_val, class_to_idx = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size, args.val_ratio, args.num_workers
    )
    num_classes = len(class_to_idx)
    print(f"Classes: {class_to_idx}")

    model = CustomCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = Path(args.out_dir) / "custom_cnn_gender.pt"
    with open(Path(args.out_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0

        for x, y in loader_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()
        train_loss = running / max(1, n)
        val_loss, val_acc = evaluate(model, loader_val, device, label_smoothing=args.label_smoothing)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping + save best
        if val_acc > best_acc + 1e-5:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "arch": "CustomCNN"
            }, best_path)
            print(f"  ✅ Saved best to: {best_path} (val_acc={val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if args.early_stop > 0 and no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch} (no improvement for {no_improve} epochs).")
                break

    print("Done.")


# ------------------------- Main -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train CUSTOM CNN from scratch (no pretrained).")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir",  type=str, default="artifacts")
    p.add_argument("--epochs",   type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--dropout",     type=float, default=0.30)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--early_stop",  type=int, default=7, help="stop if no val improvement for N epochs (0=off)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
