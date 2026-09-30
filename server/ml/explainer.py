"""
ml/explainer.py - GNNExplainer wrapper returning structured JSON for the frontend
"""
import os
import torch
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any
from torch_geometric.explain import Explainer, GNNExplainer
from server.ml import loader
from server.config import STATIC_DIR


_explainer_instance = None


def _get_explainer():
    global _explainer_instance
    if _explainer_instance is None:
        model = loader.get_model()
        if model is None:
            raise RuntimeError("Model not loaded")
        _explainer_instance = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="raw",
            ),
        )
    return _explainer_instance


def explain_node(node_id: int) -> Dict[str, Any]:
    """Run GNNExplainer and return structured explanation + image path."""
    data     = loader.get_data()
    explainer = _get_explainer()
    feat_names = loader.get_feature_names()
    tx_id = loader.get_tx_id_by_node(node_id) or node_id

    print(f"[XAI] Explainer started")
    from torch_geometric.utils import k_hop_subgraph
    # Match the model's receptive field. The model has 2 graph layers, so hops
    # beyond 2 cannot influence the prediction — they only inflate the subgraph.
    # On the rebuilt (denser) graph a 3-hop ball around a hub node reaches tens
    # of thousands of nodes and makes this endpoint time out.
    model_ = loader.get_model()
    n_hops = getattr(model_, "num_layers", 2) if model_ is not None else 2
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_id, num_hops=n_hops, edge_index=data.edge_index,
        relabel_nodes=True
    )
    sub_x = data.x[subset]
    sub_index = mapping[0].item() if mapping.numel() > 0 else 0

    explanation = explainer(sub_x, sub_edge_index, index=sub_index)
    print(f"[XAI] Explainer completed")
    node_mask_shape = list(explanation.node_mask.shape) if hasattr(explanation, "node_mask") and explanation.node_mask is not None else []
    edge_mask_shape = list(explanation.edge_mask.shape) if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None else []
    print(f"[XAI] Node mask shape: {node_mask_shape}")
    print(f"[XAI] Edge mask shape: {edge_mask_shape}")

    # ── Feature importances ────────────────────────────────────────────────
    node_mask = explanation.node_mask.cpu().numpy()
    if node_mask.ndim == 2:
        # shape [Sub_N, F] — take the row corresponding to our target node in the subgraph
        feat_importance = node_mask[sub_index]
    else:
        feat_importance = node_mask

    top_k     = min(10, len(feat_importance))
    top_idx   = np.argsort(feat_importance)[::-1][:top_k]
    top_names = [feat_names[i] if i < len(feat_names) else f"Feature_{i}" for i in top_idx]
    top_vals  = feat_importance[top_idx].tolist()

    # ── Important edges ────────────────────────────────────────────────────
    edge_mask = explanation.edge_mask.cpu().numpy()
    if sub_edge_index.numel() > 0:
        sub_src, sub_dst = sub_edge_index.cpu().numpy()
        # Map back to original global node IDs
        src = subset[sub_src].cpu().numpy()
        dst = subset[sub_dst].cpu().numpy()
    else:
        src, dst = np.array([], dtype=int), np.array([], dtype=int)

    threshold = float(np.percentile(edge_mask, 80)) if len(edge_mask) > 10 else 0.0
    imp_idx   = np.where(edge_mask >= threshold)[0]
    if len(imp_idx) > 25:
        imp_idx = np.argsort(edge_mask)[::-1][:25]

    important_edges = [
        {
            "source": int(src[i]),
            "target": int(dst[i]),
            "source_tx_id": loader.get_tx_id_by_node(int(src[i])),
            "target_tx_id": loader.get_tx_id_by_node(int(dst[i])),
            "weight": float(edge_mask[i])
        }
        for i in imp_idx
    ]
    important_nodes = list({e["source"] for e in important_edges} | {e["target"] for e in important_edges})

    # ── Generate image ─────────────────────────────────────────────────────
    img_name = f"explain_{node_id}.png"
    img_path = STATIC_DIR / img_name
    _render_explanation(node_id, important_edges, top_names, top_vals, str(img_path))

    return {
        "node_id":           node_id,
        "transaction_id":    tx_id,
        "feature_names":     top_names,
        "feature_scores":    [round(v, 6) for v in top_vals],
        "important_edges":   important_edges,
        "important_nodes":   important_nodes,
        "image_url":         f"/static/{img_name}",
        "total_features":    len(feat_importance),
        "total_edges":       len(edge_mask),
    }


def _render_explanation(node_id, edges, feat_names, feat_vals, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bg = "#0f172a"
    fig.patch.set_facecolor(bg)

    # -- Graph panel ---
    ax1.set_facecolor(bg)
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=1.5)
        node_col = ["#ef4444" if n == node_id else "#3b82f6" for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_col, node_size=180, ax=ax1, alpha=0.95)
        weights = [max(0.5, G[u][v]["weight"] * 5) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color="#94a3b8", alpha=0.7, ax=ax1, arrows=False)
        labels = {n: "TARGET" if n == node_id else str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color="white", ax=ax1)
    ax1.set_title("Transaction Neighbourhood", color="white", fontsize=12, pad=12)
    ax1.axis("off")

    # -- Feature importance panel ---
    ax2.set_facecolor(bg)
    colors = ["#ef4444" if v > 0.5 else "#3b82f6" for v in feat_vals]
    y_pos  = np.arange(len(feat_names))
    ax2.barh(y_pos, feat_vals, color=colors, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_names, color="#cbd5e1", fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("Top Risk Features", color="white", fontsize=12, pad=12)
    ax2.tick_params(colors="#cbd5e1", axis="x")
    ax2.spines[:].set_color("#334155")

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=120, facecolor=bg, bbox_inches="tight")
    plt.close()
