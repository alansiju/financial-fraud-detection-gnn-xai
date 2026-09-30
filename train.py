#!/usr/bin/env python3
"""
train.py
========
Training loop for the weighted GraphSAGE fraud detector.

What changed vs. the previous version
-------------------------------------
1. MINI-BATCH TRAINING. The old loop was full-batch: one optimiser step per
   epoch, so `EPOCHS = 200` meant the network received exactly 200 gradient
   updates. The logged run confirms the damage — validation AUC was still
   climbing (+0.04 over the last 50 epochs) when the run hit the epoch cap,
   train loss and val loss were within 5e-5 of each other, and early stopping
   never fired. That is textbook underfitting. `NeighborLoader` now gives ~200
   steps per epoch instead of 1.

2. REAL CLASS IMBALANCE HANDLING. The old loss was `FocalLoss(alpha=0.25)`,
   which multiplies positives and negatives by the *same* 0.25 — it is a
   constant rescale, not class balancing. Meanwhile `compute_pos_weight()` was
   defined and never called, and the module docstring claimed pos_weight was in
   use. Net effect on a 3.5%-positive dataset: probabilities that topped out
   around 0.65 and a median of 0.19, so a 0.5 cut flagged almost nothing.
   Both `bce` (pos_weight) and a *correct* class-aware focal loss are available.

3. MODEL SELECTION ON PR-AUC. ROC-AUC is dominated by the 96.5% majority class.
   Average precision is the metric that tracks what a fraud team cares about.

4. THE DECISION THRESHOLD IS TUNED AND SAVED. evaluate.py used to discover the
   best threshold on validation, print a confusion matrix at it, and throw it
   away — while the server hardcoded 0.5. The threshold now lives in the
   checkpoint and the server reads it.

5. EDGE WEIGHTS ARE PASSED TO THE MODEL. They were computed, stored and ignored.

6. NO FUTURE LEAKAGE. Training batches sample only over train-train edges, so a
   training node cannot aggregate features from a transaction that had not
   happened yet.

Usage
-----
    py train.py
    py train.py --loss bce --epochs 30 --batch-size 2048
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.graphsage import GraphSAGE  # noqa: E402
from graph_sampler import CSRGraph, NeighborSampleLoader  # noqa: E402

GRAPH_PATH = os.path.join(ROOT, "fraud_graph.pt")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Defaults ───────────────────────────────────────────────────────────────
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
# OneCycleLR spends the first 25% of its schedule ramping the learning rate up
# and the rest annealing it down, so the best epochs are the last ones. A short
# patience stops the run while the LR is still near its peak and the validation
# metric is at its noisiest -- the first attempt early-stopped at epoch 13 with
# its best epoch at 5, having never reached the annealing phase. Patience is
# therefore set to let the schedule finish; the best checkpoint is kept in
# memory regardless, so nothing is lost by running to the end.
PATIENCE = 30
BATCH_SIZE = 2048
NUM_NEIGHBORS = [25, 10]       # fan-out per layer while training
EVAL_NEIGHBORS = [60, 30]      # generous fan-out for a near-exact eval
RANDOM_SEED = 42

HIDDEN1, HIDDEN2, HIDDEN3 = 256, 128, 64
MLP_HIDDEN = 64
DROPOUT1 = DROPOUT2 = DROPOUT3 = 0.2
NUM_LAYERS = 2


class FocalLoss(nn.Module):
    """
    Class-aware focal loss:  -alpha_t (1 - p_t)^gamma log(p_t)

    alpha_t is `alpha` for positives and `1 - alpha` for negatives. The previous
    implementation used a single constant for both, which cancels out of the
    gradient direction entirely and does nothing about imbalance.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _bar(title: str, width: int = 66) -> None:
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}")


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found: {path}\nRun build_graph.py first.")
    print(f"[INFO] Loading graph from {path} ...")
    t0 = time.time()
    data = torch.load(path, weights_only=False)
    print(f"       Loaded in {time.time()-t0:.1f}s")
    return data


def csr_view(data, edge_mask: torch.Tensor) -> CSRGraph:
    """
    CSR adjacency over the same nodes but only the edges in `edge_mask`.

    Used to give the training loader a graph containing train-train edges only,
    so a training node cannot aggregate features from a transaction that had
    not happened yet.
    """
    if data.edge_index.numel() == 0:
        return CSRGraph(data.edge_index, data.edge_weight_norm, data.num_nodes)
    return CSRGraph(data.edge_index[:, edge_mask],
                    data.edge_weight_norm[edge_mask], data.num_nodes)


@torch.no_grad()
def predict(model, x, y, csr: CSRGraph, node_idx: torch.Tensor, device,
            neighbors, batch_size: int) -> np.ndarray:
    """
    Probabilities for `node_idx`, in that order.

    Evaluation is batched rather than one full-graph forward pass: message
    passing materialises a [num_edges, num_features] tensor, and on the rebuilt
    graph that is 3.2M x 451 floats -- far past what 6 GB of VRAM holds.
    """
    model.eval()
    loader = NeighborSampleLoader(x, y, csr, node_idx, neighbors,
                                  batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        out.append(torch.sigmoid(logits[:batch.batch_size]).squeeze(1).cpu())
    return torch.cat(out).numpy()


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Best-F1 threshold on the validation set. Returns (threshold, f1)."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--pos-weight-mode", choices=["full", "sqrt", "none"],
                    default="sqrt",
                    help="full = neg/pos (~27), sqrt = sqrt(neg/pos) (~5.2). "
                         "sqrt trades a little recall for much better calibration.")
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--no-skip", action="store_true",
                    help="disable the raw-feature residual path")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _bar("Weighted GraphSAGE Fraud Detection - Training")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}   fan-out {NUM_NEIGHBORS}")
    print(f"  Loss         : {args.loss} (pos_weight={args.pos_weight_mode})")
    print(f"  Graph layers : {args.num_layers}, skip={not args.no_skip}")

    data = load_graph(GRAPH_PATH)
    in_channels = data.x.shape[1]
    split_mode = getattr(data, "split_mode", "unknown")

    n_train = int(data.train_mask.sum())
    print(f"\n[INFO] Graph : {data.num_nodes:,} nodes | "
          f"{data.edge_index.shape[1]:,} directed edges | {in_channels} features")
    print(f"[INFO] Split : {split_mode}  "
          f"(train {n_train:,} / val {int(data.val_mask.sum()):,} / "
          f"test {int(data.test_mask.sum()):,})")

    train_idx = torch.where(data.train_mask)[0]
    val_idx = torch.where(data.val_mask)[0]

    # Training sees train-train edges only; validation may additionally use
    # train-val edges, which mirrors scoring a new transaction against history.
    train_csr = csr_view(data, data.train_edge_mask)
    val_csr = csr_view(data, data.val_edge_mask)

    print(f"[INFO] Message-passing edges: "
          f"train {int(data.train_edge_mask.sum()):,} | "
          f"val {int(data.val_edge_mask.sum()):,} | "
          f"full {data.edge_index.shape[1]:,}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = GraphSAGE(
        in_channels=in_channels,
        hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3,
        mlp_hidden=MLP_HIDDEN,
        dropout1=DROPOUT1, dropout2=DROPOUT2, dropout3=DROPOUT3,
        num_layers=args.num_layers, use_skip=not args.no_skip,
    ).to(device)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    # ── Loss ────────────────────────────────────────────────────────────────
    y_train = data.y[data.train_mask]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    ratio = n_neg / max(n_pos, 1)
    if args.pos_weight_mode == "full":
        pw = ratio
    elif args.pos_weight_mode == "sqrt":
        pw = float(np.sqrt(ratio))
    else:
        pw = 1.0
    print(f"[INFO] Train fraud: {n_pos:,}/{len(y_train):,} "
          f"({100*n_pos/len(y_train):.2f}%)  neg/pos={ratio:.1f}  "
          f"pos_weight={pw:.2f}")

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], device=device))
    else:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=WEIGHT_DECAY)

    train_loader = NeighborSampleLoader(
        data.x, data.y, train_csr, train_idx, NUM_NEIGHBORS,
        batch_size=args.batch_size, shuffle=True, seed=RANDOM_SEED,
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch, pct_start=0.25,
    )
    print(f"[INFO] {steps_per_epoch} optimiser steps/epoch  ->  "
          f"{args.epochs * steps_per_epoch:,} total "
          f"(the previous full-batch loop did {args.epochs})")

    y_val_true = data.y[val_idx].numpy().astype(int)

    # ── Train ───────────────────────────────────────────────────────────────
    _bar("Training")
    header = (f"{'Epoch':>5} | {'Loss':>8} | {'Val PR-AUC':>10} | "
              f"{'Val ROC':>8} | {'Val F1*':>8} | {'thr*':>5} | "
              f"{'sec':>5} | Status")
    print(header)
    print("-" * len(header))

    best_pr, best_state, best_meta = -1.0, None, {}
    patience_cnt = 0
    log: list[dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_seen = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            # Only the seed nodes of the batch carry a supervised label;
            # the sampled neighbours are context.
            seed = batch.batch_size
            loss = criterion(logits[:seed].squeeze(1),
                             batch.y[:seed].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * seed
            n_seen += seed

        train_loss = total_loss / max(n_seen, 1)

        probs_val = predict(model, data.x, data.y, val_csr, val_idx, device,
                            EVAL_NEIGHBORS, args.batch_size * 2)
        pr_auc = average_precision_score(y_val_true, probs_val)
        roc = roc_auc_score(y_val_true, probs_val)
        thr, f1 = tune_threshold(y_val_true, probs_val)

        status = ""
        if pr_auc > best_pr:
            best_pr = pr_auc
            patience_cnt = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_meta = {"epoch": epoch, "val_pr_auc": pr_auc,
                         "val_roc_auc": roc, "val_f1": f1, "threshold": thr}
            status = "[SAVED]"
        else:
            patience_cnt += 1

        log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_pr_auc": round(float(pr_auc), 6),
            "val_auc": round(float(roc), 6),
            "val_f1": round(float(f1), 6),
            "threshold": round(thr, 3),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(time.time() - t0, 2),
        })

        print(f"{epoch:>5} | {train_loss:>8.4f} | {pr_auc:>10.4f} | "
              f"{roc:>8.4f} | {f1:>8.4f} | {thr:>5.2f} | "
              f"{time.time()-t0:>5.1f} | {status}")

        if patience_cnt >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(no PR-AUC gain for {args.patience} epochs)")
            break

    total_time = time.time() - t_start

    # ── Save ────────────────────────────────────────────────────────────────
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    torch.save({
        "epoch": best_meta["epoch"],
        "model_state_dict": best_state,
        "val_pr_auc": best_meta["val_pr_auc"],
        "val_auc": best_meta["val_roc_auc"],
        "val_f1": best_meta["val_f1"],
        # The server reads this instead of assuming 0.5.
        "threshold": best_meta["threshold"],
        "in_channels": in_channels,
        "split_mode": split_mode,
        "feature_names": list(getattr(data, "feature_names", [])),
        "config": {
            "hidden1": HIDDEN1, "hidden2": HIDDEN2, "hidden3": HIDDEN3,
            "mlp_hidden": MLP_HIDDEN,
            "dropout1": DROPOUT1, "dropout2": DROPOUT2, "dropout3": DROPOUT3,
            "num_layers": args.num_layers, "use_skip": not args.no_skip,
        },
    }, MODEL_PATH)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    _bar("Training Complete")
    print(f"  Best epoch      : {best_meta['epoch']}")
    print(f"  Best val PR-AUC : {best_meta['val_pr_auc']:.4f}")
    print(f"  Val ROC-AUC     : {best_meta['val_roc_auc']:.4f}")
    print(f"  Val F1          : {best_meta['val_f1']:.4f} "
          f"at threshold {best_meta['threshold']:.2f}")
    print(f"  Total time      : {total_time:.1f}s")
    print(f"  Model  -> {MODEL_PATH}")
    print(f"  Log    -> {LOG_PATH}")
    print("\nNext: py evaluate.py  &&  py baselines.py")


if __name__ == "__main__":
    main()
