#!/usr/bin/env python3
"""
evaluate.py
===========
Scores the trained GraphSAGE model on the held-out test split and writes both
the plots and a machine-readable metrics row.

What changed vs. the previous version
-------------------------------------
* It used to tune the best-F1 threshold on validation, print a confusion matrix
  at that threshold, and then discard it — while the server scored everything at
  a hardcoded 0.5. The threshold now comes from the checkpoint (train.py tuned
  and saved it), so this report and the running app agree by construction.

* It never persisted its numbers anywhere. visualize.py therefore drew a chart
  captioned "Traditional ML vs GNN" that contained only Logistic Regression and
  Random Forest bars. Metrics are now appended to
  outputs/results/model_metrics.csv alongside the baselines.

* Test-set inference is batched; a full-graph forward pass no longer fits.

Outputs
-------
    outputs/results/model_metrics.csv     (GraphSAGE row, merged with baselines)
    outputs/results/gnn_test_metrics.json
    outputs/plots/{roc_curve,pr_curve,confusion_matrix,training_curves}.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.graphsage import GraphSAGE  # noqa: E402
from train import EVAL_NEIGHBORS, csr_view, predict, tune_threshold  # noqa: E402

GRAPH_PATH = os.path.join(ROOT, "fraud_graph.pt")
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "best_model.pt")
LOG_PATH = os.path.join(ROOT, "outputs", "results", "training_log.json")
PLOT_DIR = os.path.join(ROOT, "outputs", "plots")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")
BASELINE_CSV = os.path.join(RESULTS_DIR, "baseline_metrics.csv")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curves(log: list, out_path: str) -> None:
    epochs = [e["epoch"] for e in log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    ax1.set_facecolor("#0d1117")
    ax1.plot(epochs, [e["train_loss"] for e in log], color="#3498db",
             label="Train Loss")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.set_title("Training Loss", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    ax2.set_facecolor("#0d1117")
    if "val_pr_auc" in log[0]:
        ax2.plot(epochs, [e["val_pr_auc"] for e in log], color="#f1c40f",
                 label="Val PR-AUC")
    ax2.plot(epochs, [e["val_auc"] for e in log], color="#2ecc71",
             label="Val ROC-AUC")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Score", color="white")
    ax2.set_title("Validation Performance", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def write_metrics_row(row: dict) -> None:
    """Merge the GraphSAGE row into model_metrics.csv next to the baselines."""
    rows = []
    if os.path.exists(BASELINE_CSV):
        rows.extend(pd.read_csv(BASELINE_CSV).to_dict("records"))
    rows = [r for r in rows if r.get("Model") != row["Model"]]
    rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"  Metrics -> {METRICS_CSV}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")

    print("Loading graph ...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    print("Loading best model ...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = GraphSAGE(
        in_channels=ckpt.get("in_channels", data.x.shape[1]),
        hidden1=cfg.get("hidden1", 256), hidden2=cfg.get("hidden2", 128),
        hidden3=cfg.get("hidden3", 64), mlp_hidden=cfg.get("mlp_hidden", 64),
        dropout1=cfg.get("dropout1", 0.2), dropout2=cfg.get("dropout2", 0.2),
        dropout3=cfg.get("dropout3", 0.2),
        num_layers=cfg.get("num_layers", 2), use_skip=cfg.get("use_skip", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split_mode = ckpt.get("split_mode", getattr(data, "split_mode", "unknown"))
    print(f"Split mode: {split_mode}")

    val_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]
    val_csr = csr_view(data, data.val_edge_mask)
    # Test-time message passing may use the whole graph: scoring a new
    # transaction against all prior history is exactly what production does.
    full_csr = csr_view(data, torch.ones(data.edge_index.shape[1], dtype=torch.bool))

    # The threshold the model was shipped with. Recomputed here only as a check.
    thr = ckpt.get("threshold")
    if thr is None:
        print("[WARN] checkpoint has no saved threshold - retuning on validation")
        probs_val = predict(model, data.x, data.y, val_csr, val_idx, device,
                            EVAL_NEIGHBORS, 4096)
        thr, _ = tune_threshold(data.y[val_idx].numpy().astype(int), probs_val)
    print(f"Decision threshold (tuned on validation): {thr:.2f}")

    print("Evaluating on the test split ...")
    probs = predict(model, data.x, data.y, full_csr, test_idx, device,
                    EVAL_NEIGHBORS, 4096)
    y_true = data.y[test_idx].numpy().astype(int)
    preds = (probs >= thr).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    prauc = average_precision_score(y_true, probs)

    print("\n" + "=" * 46)
    print("  GraphSAGE Test Metrics")
    print("=" * 46)
    print(f"  Split      : {split_mode}")
    print(f"  Threshold  : {thr:.2f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  PR-AUC     : {prauc:.4f}")
    print(f"  Prob range : {probs.min():.4f} .. {probs.max():.4f} "
          f"(median {np.median(probs):.4f})")
    print("=" * 46)

    write_metrics_row({
        "Model": "GraphSAGE (GNN)", "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc, "PR-AUC": prauc,
    })
    with open(os.path.join(RESULTS_DIR, "gnn_test_metrics.json"), "w") as f:
        json.dump({
            "split_mode": split_mode, "threshold": float(thr),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc, "pr_auc": prauc,
            "prob_min": float(probs.min()), "prob_max": float(probs.max()),
            "prob_median": float(np.median(probs)),
            "n_test": int(len(y_true)), "n_fraud": int(y_true.sum()),
        }, f, indent=2)

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({split_mode} split)"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    p, r, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, color="green", lw=2, label=f"PR-AUC = {prauc:.3f}")
    plt.axhline(y_true.mean(), color="grey", ls="--", lw=1,
                label=f"random = {y_true.mean():.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({split_mode} split)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pr_curve.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold {thr:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
        if log:
            plot_training_curves(log, os.path.join(PLOT_DIR, "training_curves.png"))

    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
