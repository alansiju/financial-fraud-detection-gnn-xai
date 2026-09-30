#!/usr/bin/env python3
"""
graph_sampler.py
================
A dependency-free replacement for `torch_geometric.loader.NeighborLoader`.

PyG's NeighborLoader requires the compiled `pyg-lib` or `torch-sparse`
extension, and neither publishes a Windows wheel for torch 2.14 (building from
source needs a toolchain this machine does not have). Since mini-batch training
is the single biggest fix this project needs — the old full-batch loop took one
gradient step per epoch — the sampler is implemented here directly on a CSR
adjacency using numpy.

Semantics match NeighborLoader closely enough to be a drop-in:
  * `batch.x`, `batch.edge_index`, `batch.edge_attr`, `batch.y` are the induced
    subgraph in local node ids,
  * the first `batch.batch_size` rows are the seed nodes, in the order given,
    so `out[:batch_size]` are the seeds' predictions.
"""

from __future__ import annotations

import numpy as np
import torch


def _ragged_arange(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Concatenation of `range(starts[i], starts[i] + lengths[i])` for every i,
    without a Python loop.
    """
    # Zero-length entries contribute nothing and would otherwise make the
    # cumulative offsets collide with the end of the buffer.
    keep = lengths > 0
    starts = starts[keep]
    lengths = lengths[keep]

    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    offsets = np.cumsum(lengths)[:-1]
    if offsets.size:
        out[offsets] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    return np.cumsum(out)


class CSRGraph:
    """Compressed sparse row view of an undirected edge list."""

    def __init__(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 num_nodes: int):
        self.num_nodes = int(num_nodes)

        if edge_index.numel() == 0:
            self.rowptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.col = np.empty(0, dtype=np.int64)
            self.weight = np.empty(0, dtype=np.float32)
            return

        src = edge_index[0].numpy().astype(np.int64)
        dst = edge_index[1].numpy().astype(np.int64)
        w = edge_weight.numpy().astype(np.float32)

        order = np.argsort(src, kind="stable")
        src, dst, w = src[order], dst[order], w[order]

        counts = np.bincount(src, minlength=self.num_nodes)
        self.rowptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=self.rowptr[1:])
        self.col = dst
        self.weight = w

    def sample_hop(self, frontier: np.ndarray, fanout: int,
                   rng: np.random.Generator):
        """
        One hop outward. Returns (dst_nodes, src_nodes, weights) as global ids,
        where each edge points src -> dst and dst is a frontier node.
        """
        if frontier.size == 0 or self.col.size == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty(0, dtype=np.float32)

        starts = self.rowptr[frontier]
        deg = self.rowptr[frontier + 1] - starts

        if fanout is None or fanout < 0:
            take_all = np.ones(frontier.size, dtype=bool)
        else:
            take_all = deg <= fanout

        # Nodes whose whole neighbourhood fits in the budget: take every edge.
        pos_all = _ragged_arange(starts[take_all], deg[take_all])
        dst_all = np.repeat(frontier[take_all], deg[take_all])

        # Nodes over budget: sample `fanout` edges uniformly.
        sub = ~take_all
        if sub.any():
            n_sub = int(sub.sum())
            reps = np.repeat(np.arange(n_sub), fanout)
            rand = (rng.random(n_sub * fanout) * deg[sub][reps]).astype(np.int64)
            pos_sub = starts[sub][reps] + rand
            dst_sub = np.repeat(frontier[sub], fanout)
        else:
            pos_sub = np.empty(0, dtype=np.int64)
            dst_sub = np.empty(0, dtype=np.int64)

        pos = np.concatenate([pos_all, pos_sub])
        dst = np.concatenate([dst_all, dst_sub])
        return dst, self.col[pos], self.weight[pos]


class NeighborSampleLoader:
    """
    Iterable of mini-batches, each an induced multi-hop subgraph.

    Parameters
    ----------
    x, y        : full-graph node features / labels (kept on CPU)
    csr         : CSRGraph over the edges message passing may use
    input_nodes : LongTensor of seed node ids to iterate over
    fanouts     : neighbours sampled per hop, outermost hop last
                  (e.g. [25, 10] for a 2-layer model)
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, csr: CSRGraph,
                 input_nodes: torch.Tensor, fanouts: list[int],
                 batch_size: int = 2048, shuffle: bool = False,
                 seed: int = 42):
        self.x = x
        self.y = y
        self.csr = csr
        self.input_nodes = input_nodes.numpy().astype(np.int64)
        self.fanouts = list(fanouts)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        # Scratch buffer for global -> local id mapping, reused every batch.
        self._slot = np.full(csr.num_nodes, -1, dtype=np.int64)

    def __len__(self) -> int:
        return (len(self.input_nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.input_nodes
        if self.shuffle:
            idx = idx[self.rng.permutation(len(idx))]
        for start in range(0, len(idx), self.batch_size):
            yield self._build(idx[start:start + self.batch_size])

    def _build(self, seeds: np.ndarray):
        slot = self._slot
        nodes = [seeds]
        slot[seeds] = np.arange(seeds.size)
        n_nodes = seeds.size

        frontier = seeds
        e_dst, e_src, e_w = [], [], []

        for fanout in self.fanouts:
            dst, src, w = self.csr.sample_hop(frontier, fanout, self.rng)
            if dst.size == 0:
                break
            e_dst.append(dst)
            e_src.append(src)
            e_w.append(w)

            # Register newly discovered nodes and make them the next frontier.
            new = src[slot[src] < 0]
            if new.size:
                new = np.unique(new)
                slot[new] = np.arange(n_nodes, n_nodes + new.size)
                nodes.append(new)
                n_nodes += new.size
            frontier = new

        node_ids = np.concatenate(nodes) if len(nodes) > 1 else nodes[0]

        if e_dst:
            dst = np.concatenate(e_dst)
            src = np.concatenate(e_src)
            w = np.concatenate(e_w)
            local_ei = np.stack([slot[src], slot[dst]])
            edge_index = torch.from_numpy(local_ei).long()
            edge_attr = torch.from_numpy(w).float()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float)

        # Release the scratch slots we touched so the buffer stays reusable.
        slot[node_ids] = -1

        node_t = torch.from_numpy(node_ids).long()
        batch = _Batch(
            x=self.x[node_t],
            y=self.y[node_t],
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_size=int(seeds.size),
            n_id=node_t,
        )
        return batch


class _Batch:
    """Minimal stand-in for a PyG Data batch."""

    __slots__ = ("x", "y", "edge_index", "edge_attr", "batch_size", "n_id")

    def __init__(self, x, y, edge_index, edge_attr, batch_size, n_id):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch_size = batch_size
        self.n_id = n_id

    def to(self, device):
        self.x = self.x.to(device, non_blocking=True)
        self.y = self.y.to(device, non_blocking=True)
        self.edge_index = self.edge_index.to(device, non_blocking=True)
        self.edge_attr = self.edge_attr.to(device, non_blocking=True)
        return self
