#!/usr/bin/env python3
# cfllm/train_npe_seed.py
from __future__ import annotations
import os, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim

class SeedMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser("Train seed NPE")
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)  # e.g., models/npe_seed.pt
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = np.load(args.dataset)
    X = data["X"].astype(np.float32)      # (N,14)
    seeds = data["y"].astype(np.int64)    # actual seeds (labels)

    uniq = np.unique(seeds)
    seed2cls = {s:i for i,s in enumerate(uniq)}
    y = np.array([seed2cls[s] for s in seeds], dtype=np.int64)
    num_classes = len(uniq)

    N = len(X)
    idx = np.random.permutation(N)
    n_val = int(args.val_split * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    device = torch.device("cpu")
    model = SeedMLP(in_dim=X.shape[1], num_classes=num_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def run_epoch(x, t, train=True):
        bs = 64; tot_loss = 0.0; correct = 0
        model.train() if train else model.eval()
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i:i+bs], device=device)
            tb = torch.tensor(t[i:i+bs], device=device)
            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = crit(logits, tb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
            tot_loss += float(loss.detach()) * len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == tb).sum())
        return tot_loss/len(x), correct/len(x)

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(Xtr, ytr, train=True)
        va_loss, va_acc = run_epoch(Xva, yva, train=False)
        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:03d}] train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

    scripted = torch.jit.script(model.cpu())
    scripted.save(args.out)
    np.savez(os.path.splitext(args.out)[0] + "_meta.npz", uniq_seeds=uniq)
    print(f"[✓] Saved -> {args.out}")

if __name__ == "__main__":
    main()