#!/usr/bin/env python3
"""
explainability.py
=================
Generates explanations using GNNExplainer for fraud and legit predictions.
Visualizes top important features and subgraphs.

Outputs
-------
    outputs/plots/explain_fraud_node_<id>.png
    outputs/plots/explain_legit_node_<id>.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx

from torch_geometric.explain import Explainer, GNNExplainer

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.graphsage import GraphSAGE

GRAPH_PATH = os.path.join(ROOT, "fraud_graph.pt")
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "best_model.pt")
PLOT_DIR   = os.path.join(ROOT, "outputs", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


def load_model_and_data(device):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("Model or graph not found. Run train.py first.")

    print("Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False).to(device)

    print("Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    cfg = ckpt.get("config", {})
    model = GraphSAGE(
        in_channels=ckpt.get("in_channels", data.x.shape[1]),
        hidden1=cfg.get("hidden1", 256),
        hidden2=cfg.get("hidden2", 128),
        mlp_hidden=cfg.get("mlp_hidden", 64),
        dropout1=cfg.get("dropout1", 0.5),
        dropout2=cfg.get("dropout2", 0.3)
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, data


def visualize_explanation(explanation, node_idx, pred_class, out_path, max_nodes=30):
    # Get edge mask and node mask
    edge_mask = explanation.edge_mask.cpu().numpy()
    node_feat_mask = explanation.node_mask.cpu().numpy()[0] # importance of features for the node

    # Get subgraph
    src, dst = explanation.edge_index.cpu().numpy()
    
    # Filter to most important edges
    threshold = np.percentile(edge_mask, 80) if len(edge_mask) > 10 else 0
    important_idx = np.where(edge_mask >= threshold)[0]
    
    # Limit graph size for visualization
    if len(important_idx) > max_nodes:
        # Sort by importance and take top max_nodes
        top_idx = np.argsort(edge_mask)[::-1][:max_nodes]
        important_idx = top_idx

    important_src = src[important_idx]
    important_dst = dst[important_idx]
    important_weights = edge_mask[important_idx]

    # Create networkx graph
    G = nx.Graph()
    for s, d, w in zip(important_src, important_dst, important_weights):
        G.add_edge(s, d, weight=w)
        
    # Plotting
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor("#0d1117")
    
    # Left subplot: Subgraph
    ax1 = plt.subplot(121)
    ax1.set_facecolor("#0d1117")
    
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)
        
        # Color target node differently
        node_colors = []
        for n in G.nodes():
            if n == node_idx:
                node_colors.append('#e74c3c') # Target node (Red)
            else:
                node_colors.append('#3498db') # Context node (Blue)
                
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax1)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u,v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='#8ab4f8', alpha=0.6, ax=ax1)
        
    ax1.set_title(f"Important Subgraph (Node {node_idx})", color="white")
    ax1.axis('off')

    # Right subplot: Feature importance
    ax2 = plt.subplot(122)
    ax2.set_facecolor("#0d1117")
    
    top_k = 10
    top_feat_idx = np.argsort(node_feat_mask)[::-1][:top_k]
    top_feat_vals = node_feat_mask[top_feat_idx]
    
    y_pos = np.arange(top_k)
    ax2.barh(y_pos, top_feat_vals, align='center', color="#2ecc71")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Feature {i}" for i in top_feat_idx], color="white")
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance', color="white")
    ax2.set_title(f"Top {top_k} Important Features\nPrediction: {'Fraud' if pred_class==1 else 'Legit'}", color="white")
    ax2.tick_params(colors="white")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model, data = load_model_and_data(device)

    print("Setting up Explainer...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    # Find confident predictions to explain
    print("Finding target nodes...")
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).int()
    
    # We only look at test set nodes
    test_idx = torch.where(data.test_mask)[0]
    
    # Correct predictions
    correct_mask = (preds[test_idx] == data.y[test_idx])
    
    # True Positives (Confident Fraud)
    tp_mask = correct_mask & (data.y[test_idx] == 1)
    tp_indices = test_idx[tp_mask]
    
    # True Negatives (Confident Legit)
    tn_mask = correct_mask & (data.y[test_idx] == 0)
    tn_indices = test_idx[tn_mask]
    
    if len(tp_indices) > 0:
        # Sort by confidence
        tp_conf = probs[tp_indices]
        top_tp = tp_indices[torch.argsort(tp_conf, descending=True)][:3]
        
        for i, node_idx in enumerate(top_tp.cpu().numpy()):
            print(f"Explaining Fraud Node {node_idx}...")
            explanation = explainer(data.x, data.edge_index, index=int(node_idx))
            visualize_explanation(
                explanation, 
                int(node_idx), 
                pred_class=1, 
                out_path=os.path.join(PLOT_DIR, f"explain_fraud_node_{node_idx}.png")
            )
            
    if len(tn_indices) > 0:
        # Sort by confidence
        tn_conf = probs[tn_indices]
        # For legit, low probability means confident negative
        top_tn = tn_indices[torch.argsort(tn_conf, descending=False)][:3]
        
        for i, node_idx in enumerate(top_tn.cpu().numpy()):
            print(f"Explaining Legit Node {node_idx}...")
            explanation = explainer(data.x, data.edge_index, index=int(node_idx))
            visualize_explanation(
                explanation, 
                int(node_idx), 
                pred_class=0, 
                out_path=os.path.join(PLOT_DIR, f"explain_legit_node_{node_idx}.png")
            )
            
    print(f"Explanations saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
