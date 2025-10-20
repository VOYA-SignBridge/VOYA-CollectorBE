import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from torch_dataset import SignDataset


class BiGRUModel(nn.Module):
    def __init__(self, input_dim=226, hidden=256, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.fc(out)


def collate_fn(batch):
    seqs, labels, users = zip(*batch)
    seqs = torch.stack(seqs)
    labels = torch.tensor(labels, dtype=torch.long)
    return seqs, labels


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    torch.save(state, str(filepath))
    if is_best:
        best_path = out_dir / 'model_best.pth.tar'
        torch.save(state, str(best_path))


def train(args):
    # dataset
    ds = SignDataset(augment=True, max_samples=args.max_samples)
    if len(ds) == 0:
        print('No samples found under dataset/features — abort')
        return

    # build splits
    n = len(ds)
    val_n = int(n * args.val_split) if args.val_split > 0 else 0
    train_n = n - val_n
    if args.user_split:
        # split by user: group samples by user and assign users
        user_map = {}
        for idx, (_, lbl, user) in enumerate(ds.samples):
            user_map.setdefault(user, []).append(idx)
        users = list(user_map.keys())
        random.seed(args.seed)
        random.shuffle(users)
        # allocate users to train/val
        train_users = set(users[:-max(1, int(len(users)*args.val_split))])
        train_idx = [i for u in train_users for i in user_map.get(u, [])]
        val_idx = [i for u in users if u not in train_users for i in user_map.get(u, [])]
        train_dataset = torch.utils.data.Subset(ds, train_idx)
        val_dataset = torch.utils.data.Subset(ds, val_idx)
    else:
        train_dataset, val_dataset = random_split(ds, [train_n, val_n]) if val_n>0 else (ds, None)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset is not None else None

    # model
    # infer num_classes
    classes = set([lbl for _, lbl, _ in ds.samples])
    num_classes = max(classes) + 1
    model = BiGRUModel(input_dim=226, hidden=args.hidden, num_layers=args.num_layers, num_classes=num_classes, dropout=args.dropout)

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val = 0.0

    # resume
    if args.resume:
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            best_val = ckpt.get('best_val', 0.0)
            print(f'Resumed from {args.resume} at epoch {start_epoch}, best_val={best_val}')
        else:
            print('Resume checkpoint not found:', args.resume)

    # logging
    writer = SummaryWriter(log_dir=args.logdir)
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        cnt = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            cnt += 1
            if global_step % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item(), global_step)
            global_step += 1

        avg_loss = running_loss / max(1, cnt)
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f} time={elapsed:.1f}s')
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)

        # validation
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / max(1, total)
            print(f'  Val acc: {val_acc:.4f} ({correct}/{total})')
            writer.add_scalar('val/acc', val_acc, epoch)
            scheduler.step(val_acc)

        # checkpoint
        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        ckpt = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
        }
        save_checkpoint(ckpt, is_best, args.out_dir)

    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='dataset/features')
    p.add_argument('--max-samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--out-dir', default='models')
    p.add_argument('--logdir', default='runs/exp')
    p.add_argument('--resume', default='')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--user-split', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
