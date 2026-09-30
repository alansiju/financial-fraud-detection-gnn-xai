#!/usr/bin/env python3
"""
visualize.py
============
A unified script to generate presentation-ready visualizations 
from the outputs of the evaluation and baseline runs.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
PLOT_DIR = os.path.join(ROOT, "outputs", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# model_metrics.csv holds the baselines AND the GraphSAGE row that evaluate.py
# writes. The old code read baseline_metrics.csv, which never contained a GNN —
# so the chart titled "Traditional ML vs GNN" showed only Logistic Regression
# and Random Forest.
METRICS_PATH  = os.path.join(RESULTS_DIR, "model_metrics.csv")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.csv")

def plot_model_comparison():
    path = METRICS_PATH if os.path.exists(METRICS_PATH) else BASELINE_PATH
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Run baselines.py and evaluate.py first.")
        return

    df = pd.read_csv(path)
    if not df["Model"].str.contains("GraphSAGE", case=False).any():
        print("Warning: no GraphSAGE row found - run evaluate.py so the GNN "
              "appears in the comparison.")

    metrics = ["ROC-AUC", "PR-AUC", "F1", "Accuracy"]
    metrics = [m for m in metrics if m in df.columns]
    
    # Melt the dataframe for seaborn
    df_melt = pd.melt(df, id_vars=["Model"], value_vars=metrics, 
                      var_name="Metric", value_name="Score")
                      
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df_melt, palette="viridis")
    
    plt.title("Performance Comparison: Traditional ML vs GNN", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, rotation=90)
        
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "final_model_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    print("Generating unified visualizations...")
    plot_model_comparison()
    print("Done!")

if __name__ == "__main__":
    main()
