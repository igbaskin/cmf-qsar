"""
A graph neural network on the alloy slabs, as one more method beside the fields (plan item EC12).

The network is a small SchNet-type message-passing model, written in plain PyTorch (no graph
library is needed): every atom is a node, every pair of atoms nearer than 5 A is an edge, and a
filter that depends on the distance weights the message along the edge. The edge filter is a
small network on 24 Gaussians of the distance. Three message-passing layers; 32 features. The
graphs are lists of edges with distances computed once.

Two kinds of sample, as in ``run_alloy.py``:

site   The nodes are the atoms within 6 A of the point of the site (the same point and the same
       6 A as the field methods use; the slab is repeated in the plane) and one more node for the
       point itself. The output is the readout of the point node. The network gets no site kind
       and no relaxed height: only the elements and the distances.
slab   The nodes are the 12 atoms of the bare slab, with the periodic images in the plane as the
       sources of messages. The output is the mean of the readout of the 12 nodes.

Every atom starts from a learned vector for its element and the 9 tabulated properties of the
element that the references use.

Training. Rows, folds and seeds are those of the stored run of ``run_alloy.py``. In every outer
fold the training part is split in the same kind of inner folds (random, slab-out or alloy-out); a
member of the ensemble trains on four fifths and stops at the best epoch on the fifth that is
left out. Three members use three different inner folds and three seeds; the prediction is their
mean. Adam, learning rate halved when the validation error stops falling, L1 loss, at most 150
epochs. The settings were fixed in a pilot on the inner folds of one training part, not on a
test fold.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

CUTOFF = 5.0            # edge length, A
ENV_RADIUS = 6.0        # atoms in the graph of a site, A
N_RBF = 24
HIDDEN = 32
LAYERS = 3
LR = 2e-3
MAX_EPOCHS = 150
PATIENCE = 25
BATCH = 32
MEMBERS = 3


# --------------------------------------------------------------------------- #
# The graphs
# --------------------------------------------------------------------------- #
def _images(slab) -> np.ndarray:
    """The atoms of a slab with two periodic images on each side in the plane: (25 * n, 3)."""
    cell = slab.cell.array
    return np.concatenate([slab.positions + i * cell[0] + j * cell[1]
                           for i in range(-2, 3) for j in range(-2, 3)])


class Graphs:
    """The graphs of one task: nodes and edges of every sample, in long arrays with pointers.

    ``Z[node_ptr[k]:node_ptr[k+1]]`` are the elements of the nodes of sample k. An edge (c, s, d)
    sends a message from node s to node c over the distance d (both are numbers inside the
    sample); every pair nearer than CUTOFF gives two edges."""

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 readout: str):
        self.readout = readout
        self.n = len(samples)
        self.node_ptr = np.concatenate([[0], np.cumsum([len(z) for z, _, _, _ in samples])])
        self.edge_ptr = np.concatenate([[0], np.cumsum([len(c) for _, c, _, _ in samples])])
        self.Z = np.concatenate([z for z, _, _, _ in samples]).astype(np.int64)
        self.C = np.concatenate([c for _, c, _, _ in samples]).astype(np.int64)
        self.S = np.concatenate([s for _, _, s, _ in samples]).astype(np.int64)
        self.D = np.concatenate([d for _, _, _, d in samples]).astype(np.float32)

    def batch(self, idx):
        """Tensors of a batch: the elements of the nodes, the centre and the source of every edge,
        the distances, the sample of every node, the first node of every sample, the sample count."""
        import torch
        z, c, s, d, owner = [], [], [], [], []
        offset = 0
        for b, k in enumerate(idx):
            n0, n1 = self.node_ptr[k], self.node_ptr[k + 1]
            e0, e1 = self.edge_ptr[k], self.edge_ptr[k + 1]
            z.append(self.Z[n0:n1])
            c.append(self.C[e0:e1] + offset)
            s.append(self.S[e0:e1] + offset)
            d.append(self.D[e0:e1])
            owner.append(np.full(n1 - n0, b))
            offset += n1 - n0
        owner = np.concatenate(owner)
        first = np.concatenate([[0], np.cumsum(np.bincount(owner))[:-1]])
        return (torch.as_tensor(np.concatenate(z)), torch.as_tensor(np.concatenate(c)),
                torch.as_tensor(np.concatenate(s)), torch.as_tensor(np.concatenate(d)),
                torch.as_tensor(owner), torch.as_tensor(first), len(idx))


def _edges(pos_c: np.ndarray, pos_s: np.ndarray, source_index: np.ndarray):
    """Edges between the centres and the sources that are nearer than CUTOFF (not the same atom)."""
    d = np.linalg.norm(pos_c[:, None, :] - pos_s[None, :, :], axis=2)
    c, s = np.nonzero((d < CUTOFF) & (d > 1e-4))
    return c, source_index[s], d[c, s]


def with_site_kinds(graphs: Graphs, rows: Sequence[dict], first: int, kinds: Sequence[str]) -> Graphs:
    """A copy of the graphs in which the point node has the index ``first + kind`` (a vector of
    its own for every kind of site). The edges and distances are shared with the original."""
    import copy
    out = copy.copy(graphs)
    out.Z = graphs.Z.copy()
    for k, r in enumerate(rows):
        out.Z[graphs.node_ptr[k]] = first + list(kinds).index(r["site"])
    return out


def site_graphs(data, rows: Sequence[dict], ads: str, height: Dict[str, float]) -> Graphs:
    """Nodes of every site: node 0 is the point, then the atoms within ENV_RADIUS of it.

    The point has the element index len(elements)."""
    col = {e: i for i, e in enumerate(data.elements)}
    point = len(col)
    per_slab: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    samples = []
    for r in rows:
        s = r["slab"]
        if s not in per_slab:
            slab = data.slabs[s]
            top, _ = data.layers(s)
            idx = np.tile([col[e] for e in slab.get_chemical_symbols()], 25)
            per_slab[s] = (_images(slab), idx, float(slab.positions[top, 2].mean()))
        pos, idx, z_top = per_slab[s]
        centre = np.array([r["x"], r["y"], z_top + height[ads]])
        d = np.linalg.norm(pos - centre, axis=1)
        keep = np.where(d <= ENV_RADIUS)[0]
        P = np.vstack([centre[None], pos[keep]])
        z = np.concatenate([[point], idx[keep]])
        c, src, dist = _edges(P, P, np.arange(len(P)))
        samples.append((z, c, src, dist))
    return Graphs(samples, "point")


def slab_graphs(data, ids: Sequence[int]) -> Graphs:
    """The atoms of every bare slab; the sources of messages are the periodic images in the plane,
    and an image carries the vector of the atom it copies."""
    col = {e: i for i, e in enumerate(data.elements)}
    samples = []
    for k in ids:
        slab = data.slabs[k]
        n = len(slab)
        z = np.array([col[e] for e in slab.get_chemical_symbols()])
        c, src, dist = _edges(slab.positions, _images(slab), np.tile(np.arange(n), 25))
        samples.append((z, c, src, dist))
    return Graphs(samples, "mean")


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
SHELL_BINS = (2.5, 3.5, 4.5)            # A; the first three edges of the shell references
SHELL_CENTRES = (2.4, 3.0, 4.0, 4.75)   # the distance the network sees for an edge in each bin


def build_net(n_elements: int, props: np.ndarray, readout: str, hidden: int = HIDDEN,
              layers: int = LAYERS, use_emb: bool = True, use_props: bool = True,
              bins: bool = False):
    """The network. The last three options are for the ablation study (``--task ablate``):
    without the learned vector of the element, without the tabulated properties, and with every
    distance replaced by the centre of its shell (up to 2.5, 3.5, 4.5 A and beyond)."""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = readout
            self.register_buffer("props", torch.tensor(props, dtype=torch.float32))
            self.emb = nn.Embedding(props.shape[0], 16)
            self.use_emb, self.use_props, self.bins = use_emb, use_props, bins
            self.register_buffer("bin_edges", torch.tensor(SHELL_BINS, dtype=torch.float32))
            self.register_buffer("bin_centres", torch.tensor(SHELL_CENTRES, dtype=torch.float32))
            self.inp = nn.Linear(16 * use_emb + props.shape[1] * use_props, hidden)
            self.filt = nn.ModuleList([nn.Sequential(nn.Linear(N_RBF, hidden), nn.SiLU(),
                                                     nn.Linear(hidden, hidden))
                                       for _ in range(layers)])
            self.mix = nn.ModuleList([nn.Linear(hidden, hidden, bias=False)
                                      for _ in range(layers)])
            self.upd = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                                    nn.Linear(hidden, hidden))
                                      for _ in range(layers)])
            self.out = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                     nn.Linear(hidden // 2, 1))
            self.register_buffer("mu", torch.linspace(0.0, CUTOFF, N_RBF))
            self.gamma = 0.5 / (CUTOFF / (N_RBF - 1)) ** 2

        def forward(self, z, c, s, d, owner, first, n_samples):
            parts = ([self.emb(z)] if self.use_emb else []) + \
                    ([self.props[z]] if self.use_props else [])
            h = self.inp(torch.cat(parts, dim=-1))
            if self.bins:
                d = self.bin_centres[torch.bucketize(d, self.bin_edges)]
            fc = (0.5 * (torch.cos(np.pi * d / CUTOFF) + 1.0))[:, None]
            rbf = torch.exp(-self.gamma * (d[:, None] - self.mu) ** 2)
            for filt, mix, upd in zip(self.filt, self.mix, self.upd):
                msg = filt(rbf) * fc * mix(h)[s]
                m = torch.zeros_like(h).index_add_(0, c, msg)
                h = h + upd(m)
            o = self.out(h).squeeze(-1)
            if self.readout == "point":
                return o[first]
            total = torch.zeros(n_samples).index_add_(0, owner, o)
            return total / torch.bincount(owner, minlength=n_samples)

    return Net()


def train_member(net_args: dict, graphs: Graphs, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                 seed: int, hidden: int = HIDDEN, lr: float = LR, max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE):
    """Train on `tr`, stop at the best epoch on `va`. Returns (net, mean, sd, epochs, best MAE)."""
    import torch
    # The Gaussian features of the distance hold very small numbers; on a CPU the products with
    # such denormal numbers are about 30 times slower.
    torch.set_flush_denormal(True)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = build_net(hidden=hidden, **net_args)
    mean, sd = float(np.mean(y[tr])), float(np.std(y[tr]) + 1e-9)
    yt = torch.tensor((y - mean) / sd, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    best, best_state, bad, epochs = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        order = rng.permutation(tr)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(net(*graphs.batch(idx)), yt[idx])
            loss.backward()
            opt.step()
        mae = float(np.mean(np.abs(predict(net, graphs, va, mean, sd) - y[va])))
        sched.step(mae)
        epochs = epoch + 1
        if mae < best - 1e-4:
            best, bad = mae, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, mean, sd, epochs, best


def predict(net, graphs: Graphs, idx: np.ndarray, mean: float, sd: float) -> np.ndarray:
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), 256):
            out.append(net(*graphs.batch(idx[start:start + 256])).numpy())
    return np.concatenate(out) * sd + mean


# --------------------------------------------------------------------------- #
# Worker functions (a process pool runs the folds in parallel)
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def init_worker(graphs: Graphs, y: np.ndarray, net_args: dict, threads: int = 2):
    import torch
    torch.set_num_threads(threads)
    _STATE.update(graphs=graphs, y=y, net_args=net_args)


def fit_fold(job: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]):
    """(fold number, member number, training rows, validation rows, test rows) -> predictions."""
    fold, member, tr, va, te = job
    t0 = time.time()
    net, mean, sd, epochs, best = train_member(
        _STATE["net_args"], _STATE["graphs"], _STATE["y"], tr, va, seed=1000 * fold + member)
    return fold, member, te, predict(net, _STATE["graphs"], te, mean, sd), epochs, best, \
        time.time() - t0


def init_ablation(variants: Dict[str, Tuple[Graphs, dict]], y: np.ndarray, threads: int = 1):
    """Workers of the ablation study: ``variants`` maps a name to (graphs, net_args)."""
    import torch
    torch.set_num_threads(threads)
    _STATE.update(variants=variants, y=y)


def fit_variant(job):
    """(variant, fold number, member, training rows, validation rows, test rows) -> predictions.

    The seed depends on the fold and the member only, so that the variants are paired."""
    name, fold, member, tr, va, te = job
    t0 = time.time()
    graphs, net_args = _STATE["variants"][name]
    net, mean, sd, epochs, best = train_member(net_args, graphs, _STATE["y"], tr, va,
                                               seed=1000 * fold + member + 1)
    return name, fold, member, te, predict(net, graphs, te, mean, sd), epochs, best, \
        time.time() - t0
