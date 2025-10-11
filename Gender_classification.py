import argparse, json, os, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders(data_dir, batch_size=32, img_size=224, val_ratio=0.15):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # ถ้ามี data/val จะใช้เป็น val; ไม่งั้น split จาก train
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        ds_train = datasets.ImageFolder(str(train_dir), transform=tfm_train)
        ds_val   = datasets.ImageFolder(str(val_dir),   transform=tfm_eval)
    else:
        ds_full = datasets.ImageFolder(str(train_dir), transform=tfm_train)
        n_total = len(ds_full)
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        ds_train, ds_val = random_split(ds_full, [n_train, n_val])

        # ให้ val ใช้ tfm_eval
        ds_val.dataset.transform = tfm_eval

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # class_to_idx (จาก ImageFolder) — ถ้าใช้ random_split ให้ย้อนกลับไปที่ ds_train.dataset
    if isinstance(ds_train, torch.utils.data.Subset):
        class_to_idx = ds_train.dataset.class_to_idx
    else:
        class_to_idx = ds_train.class_to_idx
    return loader_train, loader_val, class_to_idx

def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return loss_sum/total, correct/total

def train(args):
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader_train, loader_val, class_to_idx = get_dataloaders(args.data_dir, args.batch_size, args.img_size, args.val_ratio)

    model = build_model(num_classes=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc, best_path = 0.0, Path(args.out_dir) / "gender_resnet18.pt"
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0
        n = 0
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = run_loss / max(1, n)
        val_loss, val_acc = evaluate(model, loader_val, device)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
            }, best_path)
            print(f"  ✅ Saved best model to {best_path} (val_acc={val_acc*100:.2f}%)")

    # บันทึก mapping class_to_idx เผื่อใช้นอก PyTorch
    with open(Path(args.out_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train gender classifier (ResNet18)")
    p.add_argument("--data_dir", type=str, default="data", help="path to data folder containing train/ and optional val/")
    p.add_argument("--out_dir", type=str, default="artifacts", help="where to save model")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()
    train(args)
