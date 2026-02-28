import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

try:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit  # type: ignore
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore
    StratifiedShuffleSplit = None  # type: ignore

from torch_dataset import SignDataset


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if float(d) != 0.0 else 0.0


def _compute_prf1(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, object]:
    # Basic, dependency-free per-class precision/recall/F1.
    # Returns per_class list aligned by class index (0..num_classes-1).
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes

    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes:
            support[t] += 1
        if t == p and 0 <= t < num_classes:
            tp[t] += 1
        else:
            if 0 <= p < num_classes:
                fp[p] += 1
            if 0 <= t < num_classes:
                fn[t] += 1

    per_class = []
    macro_f1_sum = 0.0
    macro_count = 0
    weighted_f1_sum = 0.0
    total_support = sum(support)
    for c in range(num_classes):
        precision = _safe_div(tp[c], tp[c] + fp[c])
        recall = _safe_div(tp[c], tp[c] + fn[c])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class.append(
            {
                'class': int(c),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(support[c]),
                'tp': int(tp[c]),
                'fp': int(fp[c]),
                'fn': int(fn[c]),
            }
        )
        # macro counts all classes equally (even if support==0, f1 will be 0)
        macro_f1_sum += float(f1)
        macro_count += 1
        weighted_f1_sum += float(f1) * float(support[c])

    macro_f1 = _safe_div(macro_f1_sum, max(1, macro_count))
    weighted_f1 = _safe_div(weighted_f1_sum, max(1, total_support))
    return {
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class,
    }


def _confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return

    def close(self):
        return


class BiGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        pooling: str = 'last',
    ):
        super().__init__()
        self.pooling = pooling
        self.rnn = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        enc_dim = hidden * 2
        if pooling == 'attn':
            self.attn = nn.Sequential(
                nn.Linear(enc_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        else:
            self.attn = None

        self.fc = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def _last_valid_pool(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        eps = 1e-8
        valid = (x.abs().sum(dim=-1) > eps)  # (B,T)
        idx = valid.long().sum(dim=1) - 1
        idx = idx.clamp(min=0)
        return out[torch.arange(out.shape[0], device=out.device), idx]

    def _attn_pool(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        valid = (x.abs().sum(dim=-1) > eps)  # (B,T)
        scores = self.attn(out).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~valid, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (out * w).sum(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.pooling == 'mean':
            pooled = out.mean(dim=1)
        elif self.pooling == 'attn':
            pooled = self._attn_pool(out, x)
        else:
            pooled = self._last_valid_pool(out, x)
        return self.fc(pooled)


def build_label_mapping(raw_labels: List[int]) -> Tuple[Dict[int, int], List[int]]:
    uniq = sorted(set(int(x) for x in raw_labels))
    raw_to_out = {raw: i for i, raw in enumerate(uniq)}
    return raw_to_out, uniq


def ensure_all_classes_in_val(
    train_idx: List[int],
    val_idx: List[int],
    y: List[int],
    groups: List[str],
) -> Tuple[List[int], List[int]]:
    all_classes = set(y)
    val_classes = set(y[i] for i in val_idx)
    missing = list(all_classes - val_classes)
    if not missing:
        return train_idx, val_idx

    # Move entire groups from train -> val until we cover missing classes (best-effort)
    group_to_indices: Dict[str, List[int]] = {}
    for i in train_idx:
        group_to_indices.setdefault(groups[i], []).append(i)

    moved = set()
    for cls in missing:
        for g, idxs in group_to_indices.items():
            if g in moved:
                continue
            if any(y[i] == cls for i in idxs):
                val_idx.extend(idxs)
                train_idx = [i for i in train_idx if i not in set(idxs)]
                moved.add(g)
                break

    return train_idx, val_idx


def stratified_group_split(y: List[int], groups: List[str], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    if val_split <= 0:
        idx = list(range(len(y)))
        return idx, []

    if StratifiedShuffleSplit is None or StratifiedGroupKFold is None:
        # Minimal fallback without sklearn
        rng = random.Random(seed)
        idx = list(range(len(y)))
        rng.shuffle(idx)
        val_n = max(1, int(round(len(idx) * val_split)))
        return idx[val_n:], idx[:val_n]

    # If no usable grouping info, fall back to stratified shuffle split.
    if len(set(groups)) < 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        X_dummy = np.zeros((len(y), 1), dtype=np.float32)
        y_arr = np.asarray(y)
        tr, va = next(splitter.split(X_dummy, y_arr))
        return tr.tolist(), va.tolist()

    n_splits = int(round(1.0 / max(min(val_split, 0.5), 0.05)))
    n_splits = max(2, min(10, n_splits))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1), dtype=np.float32)
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups)
    try:
        train_idx, val_idx = next(sgkf.split(X_dummy, y_arr, groups=g_arr))
        return train_idx.tolist(), val_idx.tolist()
    except Exception:
        # Fallback when grouping constraints make stratified split impossible
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        tr, va = next(splitter.split(X_dummy, y_arr))
        return tr.tolist(), va.tolist()


def compute_norm_stats(dataset: SignDataset, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    sums = None
    sums2 = None
    count = 0
    for i in indices:
        x, _y, _g = dataset[i]
        arr = x.numpy().astype(np.float64)  # (T,D)
        if arr.ndim != 2:
            continue
        if sums is None:
            sums = arr.sum(axis=0)
            sums2 = (arr * arr).sum(axis=0)
        else:
            sums += arr.sum(axis=0)
            sums2 += (arr * arr).sum(axis=0)
        count += arr.shape[0]

    if sums is None or count == 0:
        raise RuntimeError('Cannot compute normalization stats: empty training subset')

    mean = sums / float(count)
    var = (sums2 / float(count)) - (mean * mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    torch.save(state, str(filepath))
    if is_best:
        best_path = out_dir / 'model_best.pth.tar'
        torch.save(state, str(best_path))


def train(args):
    # dataset (hierarchy-aware)
    train_augment = (not args.no_augment)
    ds = SignDataset(
        features_root=args.data_root,
        labels_csv=args.labels_csv,
        language=args.language,
        dialect=args.dialect,
        augment=train_augment,
        max_samples=args.max_samples,
        expected_dim=args.expected_dim,
    )
    if len(ds) == 0:
        print('No samples found under dataset/features — abort')
        return

    # Filter invalid labels
    valid_indices = [i for i, s in enumerate(ds.samples) if int(s.label) >= 0]
    if not valid_indices:
        print('No valid-labeled samples found (labels.csv missing or class_idx not set) — abort')
        return

    # Split (stratified by label, grouped by session/user)
    y_full = [int(s.label) for s in ds.samples]
    groups_full = [str(s.session_id or s.user or '') for s in ds.samples]

    y_raw = [y_full[i] for i in valid_indices]
    groups = [groups_full[i] for i in valid_indices]

    train_rel, val_rel = stratified_group_split(y_raw, groups, args.val_split, args.seed)
    # Map relative indices to absolute dataset indices
    train_idx = [valid_indices[i] for i in train_rel]
    val_idx = [valid_indices[i] for i in val_rel]
    train_idx, val_idx = ensure_all_classes_in_val(train_idx, val_idx, y_full, groups_full)

    # Label mapping -> contiguous outputs
    raw_to_out, out_to_raw = build_label_mapping([int(ds.samples[i].label) for i in train_idx + val_idx])

    # Normalization (train only), saved into checkpoint
    norm_mean = None
    norm_std = None
    if not args.no_normalize:
        ds_for_stats = SignDataset(
            features_root=args.data_root,
            labels_csv=args.labels_csv,
            language=args.language,
            dialect=args.dialect,
            augment=False,
            max_samples=args.max_samples,
            expected_dim=args.expected_dim,
        )
        norm_mean, norm_std = compute_norm_stats(ds_for_stats, train_idx)
        ds.set_normalization(norm_mean, norm_std)

    train_dataset = Subset(ds, train_idx)
    val_dataset = Subset(ds, val_idx) if len(val_idx) else None

    def collate_fn(batch):
        seqs, labels, _groups = zip(*batch)
        seqs = torch.stack(seqs)
        labels = torch.tensor([raw_to_out[int(l)] for l in labels], dtype=torch.long)
        return seqs, labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset is not None else None

    input_dim = int(ds.expected_dim if ds.expected_dim is not None else (ds.input_dim or 126))
    if args.expected_dim is not None and input_dim != int(args.expected_dim):
        input_dim = int(args.expected_dim)

    num_classes = len(out_to_raw)
    model = BiGRUModel(
        input_dim=input_dim,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        pooling=args.pooling,
    )

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val = -1.0
    bad_epochs = 0

    metrics_history: List[Dict[str, object]] = []

    # resume
    if args.resume:
        if os.path.exists(args.resume):
            try:
                ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            except TypeError:  # older torch
                ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            best_val = ckpt.get('best_val', ckpt.get('best_val_acc', -1.0))
            # load normalization if available
            if not args.no_normalize:
                nm = ckpt.get('norm_mean')
                ns = ckpt.get('norm_std')
                if nm is not None and ns is not None:
                    ds.set_normalization(np.asarray(nm, dtype=np.float32), np.asarray(ns, dtype=np.float32))
            print(f'Resumed from {args.resume} at epoch {start_epoch}, best_val={best_val}')
        else:
            print('Resume checkpoint not found:', args.resume)

    # logging
    writer = SummaryWriter(log_dir=args.logdir) if SummaryWriter is not None else _NullWriter()
    global_step = 0

    # Export label_map.json aligned with output indices
    label_map: Dict[str, Dict] = {}

    # Prefer dataset/labels.csv metadata when available
    labels_by_uid: Dict[str, Dict] = {}
    try:
        import csv

        p = Path(args.labels_csv)
        if p.exists():
            with p.open('r', encoding='utf-8', newline='') as f:
                r = csv.DictReader(f)
                for row in r:
                    cu = (row.get('class_uid') or '').strip()
                    if cu:
                        labels_by_uid[cu] = row
    except Exception:
        labels_by_uid = {}

    raw_to_any_meta: Dict[int, Dict] = {}
    for s in ds.samples:
        if int(s.label) >= 0 and int(s.label) not in raw_to_any_meta:
            raw_to_any_meta[int(s.label)] = {
                'class_uid': s.class_uid,
                'slug': s.slug,
                'label_original': s.label_original,
            }

    for out_idx, raw_label in enumerate(out_to_raw):
        m = raw_to_any_meta.get(int(raw_label), {})
        row = labels_by_uid.get(m.get('class_uid', ''), {})
        label_map[str(out_idx)] = {
            'out_idx': int(out_idx),
            'class_idx': int(raw_label),
            'class_uid': m.get('class_uid', ''),
            'slug': (row.get('slug') or m.get('slug') or ''),
            'label_original': (row.get('label_original') or m.get('label_original') or ''),
            'language': (row.get('language') or args.language),
            'dialect': (row.get('dialect') or args.dialect),
        }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, 'label_map.json').write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding='utf-8')

    for epoch in range(start_epoch, args.epochs):
        # Ensure augmentation is only enabled for training.
        ds.augment = train_augment
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
        val_loss = None
        if val_loader is not None:
            ds.augment = False
            model.eval()
            correct = 0
            total = 0
            vloss_sum = 0.0
            vloss_cnt = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    vloss_sum += float(loss.item())
                    vloss_cnt += 1
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / max(1, total)
            val_loss = vloss_sum / max(1, vloss_cnt)
            print(f'  Val acc: {val_acc:.4f} ({correct}/{total})')
            writer.add_scalar('val/acc', val_acc, epoch)
            if val_loss is not None:
                writer.add_scalar('val/loss', float(val_loss), epoch)
            scheduler.step(val_acc)

            # Restore train-time augmentation for next epoch.
            ds.augment = train_augment

        lr = None
        try:
            lr = float(optimizer.param_groups[0].get('lr', 0.0))
        except Exception:
            lr = None

        metrics_history.append(
            {
                'epoch': int(epoch + 1),
                'train_loss': float(avg_loss),
                'val_loss': float(val_loss) if val_loss is not None else None,
                'val_acc': float(val_acc),
                'lr': lr,
                'time_sec': float(elapsed),
            }
        )

        # checkpoint + early stopping
        improved = val_acc > (best_val + float(args.min_delta))
        is_best = improved
        if improved:
            best_val = val_acc
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Make checkpoint portable across numpy versions by avoiding pickled numpy arrays.
        norm_mean_out = norm_mean.tolist() if isinstance(norm_mean, np.ndarray) else norm_mean
        norm_std_out = norm_std.tolist() if isinstance(norm_std, np.ndarray) else norm_std

        ckpt = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
            'best_val_acc': best_val,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'pooling': args.pooling,
            'raw_to_out': raw_to_out,
            'out_to_raw': out_to_raw,
            'norm_mean': norm_mean_out,
            'norm_std': norm_std_out,
            'data': {
                'features_root': args.data_root,
                'labels_csv': args.labels_csv,
                'language': args.language,
                'dialect': args.dialect,
                'expected_dim': args.expected_dim,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'val_split': args.val_split,
                'grouping': 'session_id_or_user',
            },
        }
        save_checkpoint(ckpt, is_best, args.out_dir)

        if bad_epochs >= int(args.patience):
            print('Early stopping triggered.')
            break

    writer.close()

    # Persist metrics history to disk
    try:
        Path(args.out_dir, 'metrics_history.json').write_text(
            json.dumps(metrics_history, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    except Exception:
        pass

    # Final evaluation on validation set using best checkpoint
    if val_loader is not None:
        try:
            best_path = Path(args.out_dir, 'model_best.pth.tar')
            if best_path.exists():
                try:
                    best_ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
                except TypeError:
                    best_ckpt = torch.load(str(best_path), map_location=device)
                model.load_state_dict(best_ckpt['state_dict'])
            model.eval()

            # Disable augmentation for evaluation metrics.
            ds.augment = False

            y_true: List[int] = []
            y_pred: List[int] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    logits = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy().tolist()
                    yb_list = yb.cpu().numpy().tolist()
                    y_true.extend([int(v) for v in yb_list])
                    y_pred.extend([int(v) for v in preds])

            num_classes_eval = int(num_classes)
            cm = _confusion_matrix(y_true, y_pred, num_classes_eval)

            # Try sklearn for richer report, fall back to basic metrics.
            report = None
            try:
                from sklearn.metrics import classification_report  # type: ignore

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes_eval)),
                    output_dict=True,
                    zero_division=0,
                )
            except Exception:
                report = None

            basic = _compute_prf1(y_true, y_pred, num_classes_eval)
            overall_acc = _safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), max(1, len(y_true)))

            val_report = {
                'val_samples': int(len(y_true)),
                'val_acc': float(overall_acc),
                'macro_f1': float(basic.get('macro_f1', 0.0)),
                'weighted_f1': float(basic.get('weighted_f1', 0.0)),
                'per_class': basic.get('per_class', []),
                'confusion_matrix': cm,
                'sklearn_report': report,
            }

            Path(args.out_dir, 'val_report.json').write_text(
                json.dumps(val_report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='dataset/features')
    p.add_argument('--labels-csv', default='dataset/labels.csv')
    p.add_argument('--language', default='vn')
    p.add_argument('--dialect', default='common')
    p.add_argument('--max-samples', type=int, default=0)
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
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--pooling', choices=['mean', 'last', 'attn'], default='last')
    p.add_argument('--expected-dim', type=int, default=126)
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
