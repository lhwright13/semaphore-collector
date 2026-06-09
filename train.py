"""Pretrain on synthetic semaphore, evaluate on real clips.

Milestone 1 - whole-phrase classification: each prompt ("hello", "thank you",
...) is one class. This matches the data the collector produces and what synth.py
generates. Later we swap the classification head for CTC to do open-vocabulary
continuous (cursive) transcription.

Flow:
  1. Build the phrase vocabulary from prompts.txt (shared by synth + real).
  2. Pretrain on the infinite synthetic stream (many examples per phrase).
  3. Evaluate on real clips as a held-out test set -> sim-to-real accuracy.

Runs on Apple Silicon MPS. --device cpu for debugging. --overfit-real also
trains on the real clips afterward as a capacity sanity check (will memorize,
since there's ~1 example per phrase - not a generalization measure).
"""
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from features import normalize_sequence
from synth import stream
from prompts import load_prompts
from model import SemaphoreRecognizer


class SyntheticDataset(IterableDataset):
    """Infinite synthetic stream. IterableDataset is the correct idiom for an
    unbounded stream (no index/length hack)."""
    def __init__(self, labels, label_to_idx, rng):
        self.labels = list(labels)
        self.label_to_idx = label_to_idx
        self.rng = rng

    def __iter__(self):
        for raw, text in stream(self.labels, self.rng):
            feat = normalize_sequence(raw)
            yield torch.from_numpy(feat), self.label_to_idx[text]


class RealDataset(Dataset):
    """Real clips from the collector, labeled by full phrase."""
    def __init__(self, label_to_idx, data_dir='data/clips'):
        self.label_to_idx = label_to_idx
        self.items = []  # (npy_path, label_idx, label_text)
        skipped = []
        for npy_path in sorted(glob.glob(f'{data_dir}/*.npy')):
            meta_path = npy_path.replace('.npy', '.json')
            if not Path(meta_path).exists():
                continue
            label = json.load(open(meta_path))['label']
            if label in label_to_idx:
                self.items.append((npy_path, label_to_idx[label], label))
            else:
                skipped.append(label)
        if skipped:
            print(f"  (skipped {len(skipped)} clips with labels not in prompts.txt: {set(skipped)})")

    def __getitem__(self, idx):
        npy_path, label_idx, _ = self.items[idx]
        feat = normalize_sequence(np.load(npy_path).astype(np.float32))
        return torch.from_numpy(feat), label_idx

    def __len__(self):
        return len(self.items)


def collate_pad(batch):
    """Pad variable-length sequences to the longest in the batch."""
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - len(s))) for s in seqs]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def train_step(model, batch, device, opt, criterion):
    seqs, labels = batch
    seqs, labels = seqs.to(device), labels.to(device)
    opt.zero_grad()
    loss = criterion(model(seqs), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        preds = model(seqs).argmax(-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / total if total else 0.0


@torch.no_grad()
def show_predictions(model, dataset, idx_to_label, device):
    model.eval()
    print("\n  real clip predictions (true -> predicted):")
    for i in range(len(dataset)):
        seq, label_idx = dataset[i]
        pred = model(seq.unsqueeze(0).to(device)).argmax(-1).item()
        mark = "ok " if pred == label_idx else "  X"
        print(f"    {mark} {idx_to_label[label_idx]:<12} -> {idx_to_label[pred]}")
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='mps', choices=['mps', 'cpu', 'cuda'])
    p.add_argument('--synth-steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--overfit-real', action='store_true',
                   help='also train on real clips afterward (capacity sanity check)')
    p.add_argument('--ckpt-dir', default='checkpoints')
    args = p.parse_args()

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    print(f"Device: {device}")
    Path(args.ckpt_dir).mkdir(exist_ok=True)

    # Shared phrase vocabulary
    labels = [label for label, _ in load_prompts()]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    print(f"Classes ({len(labels)}): {labels}")

    model = SemaphoreRecognizer(num_classes=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Synthetic pretraining
    print(f"\n=== Synthetic pretraining ({args.synth_steps} steps) ===")
    synth_ds = SyntheticDataset(labels, label_to_idx, np.random.default_rng(0))
    synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, collate_fn=collate_pad)
    synth_iter = iter(synth_loader)
    model.train()
    for step in range(args.synth_steps):
        loss = train_step(model, next(synth_iter), device, opt, criterion)
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1:5d}  loss={loss:.4f}")

    # Evaluate sim-to-real on the real clips (held out - never trained on)
    real_ds = RealDataset(label_to_idx)
    if len(real_ds):
        real_loader = DataLoader(real_ds, batch_size=args.batch_size, collate_fn=collate_pad)
        acc = evaluate(model, real_loader, device)
        print(f"\n=== Sim-to-real on {len(real_ds)} real clips: accuracy = {acc:.1%} ===")
        show_predictions(model, real_ds, idx_to_label, device)

        if args.overfit_real:
            print("\n=== Overfit sanity check on real clips ===")
            for step in range(300):
                loss = train_step(model, next(iter(real_loader)), device, opt, criterion)
                if (step + 1) % 50 == 0:
                    print(f"  step {step + 1:4d}  loss={loss:.4f}  acc={evaluate(model, real_loader, device):.1%}")
    else:
        print("\nNo real clips found - collect some with collector.py to evaluate.")

    torch.save(model.state_dict(), f"{args.ckpt_dir}/final.pth")
    print(f"\nSaved {args.ckpt_dir}/final.pth")


if __name__ == "__main__":
    main()
