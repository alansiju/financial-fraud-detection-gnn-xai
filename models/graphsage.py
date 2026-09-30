#!/usr/bin/env python3
"""
models/graphsage.py
===================
Weighted GraphSAGE with a residual node-feature path, for binary fraud
classification on the IEEE-CIS transaction graph.

Architecture
------------
                      ┌─ node path ─────────────────────────┐
    x ────────────────┤  MLP: in -> h1 -> h2, BN + ReLU     ├──┐
      │               └─────────────────────────────────────┘  │
      │                                                        │
      └─ graph path ─ GraphConv(in, h1) + BN + ReLU + Drop      │
                      GraphConv(h1, h2) + BN + ReLU + Drop ─────┤
                      (optional 3rd layer)                      │
                                                                ▼
                                             concat  →  Linear(2*h2, mlp_hidden)
                                                     →  ReLU + Dropout
                                                     →  Linear(mlp_hidden, 1)  →  logit

Why it differs from the previous version
----------------------------------------
1. `GraphConv` replaces `SAGEConv` because it accepts `edge_weight`. The old
   model ignored `data.edge_attr` entirely, so the carefully built additive
   edge weights (card1=3.0, DeviceType=1.0, ...) had no effect on anything.
   GraphConv with aggr='mean' computes  W1·x_i + W2·mean_j(e_ij · x_j),
   i.e. exactly GraphSAGE-mean with weighted neighbours.

2. The node path is new. Roughly 70% of uploaded transactions attach to no
   neighbour at all, and for those the old model was a 3-layer stack of
   near-identity aggregations. The skip connection guarantees a node's own
   features always reach the classifier at full strength.

3. Two conv layers by default instead of three — three rounds of mean
   aggregation over this graph oversmooths.

Usage
-----
    model = GraphSAGE(in_channels=X.shape[1])
    logits = model(data.x, data.edge_index, data.edge_weight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class GraphSAGE(nn.Module):
    """
    Parameters
    ----------
    in_channels : int    number of input node features
    hidden1     : int    width after the first graph layer
    hidden2     : int    width after the second graph layer
    hidden3     : int    width after the optional third graph layer
    mlp_hidden  : int    width of the classification head
    dropout1/2/3: float  dropout after each graph layer
    num_layers  : int    2 or 3 graph convolution layers
    use_skip    : bool   enable the raw-feature residual path
    aggr        : str    neighbourhood aggregation ('mean' recommended)
    """

    def __init__(
        self,
        in_channels: int = 432,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        mlp_hidden: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.3,
        dropout3: float = 0.3,
        num_layers: int = 2,
        use_skip: bool = True,
        aggr: str = "mean",
    ) -> None:
        super().__init__()

        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.num_layers = int(num_layers)
        self.use_skip = bool(use_skip)

        # ── Graph path ──────────────────────────────────────────────────────
        self.conv1 = GraphConv(in_channels, hidden1, aggr=aggr)
        self.bn1 = nn.BatchNorm1d(hidden1)

        self.conv2 = GraphConv(hidden1, hidden2, aggr=aggr)
        self.bn2 = nn.BatchNorm1d(hidden2)

        if self.num_layers >= 3:
            self.conv3 = GraphConv(hidden2, hidden3, aggr=aggr)
            self.bn3 = nn.BatchNorm1d(hidden3)
            graph_out = hidden3
        else:
            self.conv3 = None
            self.bn3 = None
            graph_out = hidden2

        # ── Node path (operates on raw features, ignores the graph) ────────
        # A two-layer MLP, not a single projection. Gradient-boosted trees reach
        # PR-AUC 0.52 on these same 451 features while a one-layer projection
        # into 128 units reached 0.41 -- the node pathway simply did not have
        # the capacity to compete on tabular data, so the graph had nothing good
        # to add context to. The graph path supplies neighbourhood context; this
        # path has to be a strong tabular model in its own right.
        if self.use_skip:
            self.skip_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(hidden1, graph_out),
                nn.BatchNorm1d(graph_out),
                nn.ReLU(),
            )
            head_in = graph_out * 2
        else:
            self.skip_mlp = None
            head_in = graph_out

        # ── Classification head ─────────────────────────────────────────────
        self.fc1 = nn.Linear(head_in, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        mods = [self.fc1, self.fc2]
        if self.skip_mlp is not None:
            mods.extend(m for m in self.skip_mlp if isinstance(m, nn.Linear))
        for m in mods:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : [N, in_channels]
        edge_index  : [2, E]  bidirectional
        edge_weight : [E] or None. None means every edge counts as 1.0.

        Returns
        -------
        logits : [N, 1] raw scores. sigmoid() for probabilities.
        """
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.bn1(h))
        h = F.dropout(h, p=self.dropout1, training=self.training)

        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(self.bn2(h))
        h = F.dropout(h, p=self.dropout2, training=self.training)

        if self.conv3 is not None:
            h = self.conv3(h, edge_index, edge_weight)
            h = F.relu(self.bn3(h))
            h = F.dropout(h, p=self.dropout3, training=self.training)

        if self.skip_mlp is not None:
            h = torch.cat([h, self.skip_mlp(x)], dim=1)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=self.dropout2, training=self.training)
        return self.fc2(h)

    # ── Helpers ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """Fraud probability in [0, 1] for every node. Shape [N]."""
        self.eval()
        return torch.sigmoid(self.forward(x, edge_index, edge_weight)).squeeze(1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"GraphSAGE(layers={self.num_layers}, skip={self.use_skip}, "
            f"conv1={self.conv1}, conv2={self.conv2}, "
            f"conv3={self.conv3}, fc1={self.fc1}, fc2={self.fc2})"
        )
