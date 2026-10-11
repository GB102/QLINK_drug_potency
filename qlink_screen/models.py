"""Models under comparison. Every hybrid shares one interface so only the circuit differs.

Hybrid head (identical across all quantum models):
    compact features (16) -> Linear(16,4) -> (pi/2)*tanh -> circuit -> <Z> on 4 data qubits -> Linear(4,out)
Circuit weights are initialized U(0, 2*pi), the team's original convention, for every circuit family.
`mlp_bneck` is the same graph with the circuit replaced by a classical 4->4 layer, so a quantum/classical
difference cannot be attributed to the 4-wide bottleneck alone.
"""
import numpy as np
import torch
import torch.nn as nn

from . import circuits
from .quantum import CircuitLayer

ANGLE_SCALE = np.pi / 2


class Hybrid(nn.Module):
    def __init__(self, model_id, n_blocks, in_dim=16, n_out=1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        qc, xs, ws, obs = circuits.build(model_id, n_blocks)
        self.circuit = qc
        self.model_id, self.n_blocks = model_id, n_blocks
        self.pre = nn.Linear(in_dim, len(xs))
        self.q = CircuitLayer(qc, xs, ws, len(obs))
        self.weights = nn.Parameter(torch.rand(len(ws), generator=g) * 2 * np.pi)
        self.post = nn.Linear(len(obs), n_out)

    def quantum_out(self, x):
        return self.q(torch.tanh(self.pre(x)) * ANGLE_SCALE, self.weights)

    def forward(self, x):
        return self.post(self.quantum_out(x))

    def quantum_parameters(self):
        return [self.weights]

    def classical_parameters(self):
        return list(self.pre.parameters()) + list(self.post.parameters())


class MLPBottleneck(nn.Module):
    """Classical control with the same 4-wide bottleneck and bounded activations as the hybrids."""

    def __init__(self, in_dim=16, n_out=1, width=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.pre = nn.Linear(in_dim, width)
        self.mid = nn.Linear(width, width)
        self.post = nn.Linear(width, n_out)

    def forward(self, x):
        h = torch.tanh(self.pre(x)) * ANGLE_SCALE
        return self.post(torch.tanh(self.mid(h)))

    def quantum_parameters(self):
        return []

    def classical_parameters(self):
        return list(self.parameters())


class MLP(nn.Module):
    """Small classical MLP on the same compact features: controls for the input bottleneck."""

    def __init__(self, in_dim=16, n_out=1, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_out))

    def forward(self, x):
        return self.net(x)

    def quantum_parameters(self):
        return []

    def classical_parameters(self):
        return list(self.parameters())


class DNNfp(nn.Module):
    """The team's original DNN architecture, on variance-filtered PubChem fingerprints."""

    def __init__(self, in_dim, n_out=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for a, b in [(in_dim, 512), (512, 256), (256, 64)]:
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(0.3)]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(64, n_out)

    def forward(self, x):
        return self.out(self.body(x))

    def quantum_parameters(self):
        return []

    def classical_parameters(self):
        return list(self.parameters())


QUANTUM_IDS = list(circuits.SPECS)


def build_model(model_id, n_blocks=4, in_dim=16, n_out=1, seed=0):
    if model_id in QUANTUM_IDS:
        return Hybrid(model_id, n_blocks, in_dim, n_out, seed)
    if model_id == "mlp_bneck":
        return MLPBottleneck(in_dim, n_out, seed=seed)
    if model_id == "mlp":
        return MLP(in_dim, n_out, seed=seed)
    if model_id == "dnn_fp":
        return DNNfp(in_dim, n_out, seed=seed)
    raise ValueError(model_id)


def count_parameters(model):
    q = sum(p.numel() for p in model.quantum_parameters())
    return {"total": sum(p.numel() for p in model.parameters() if p.requires_grad), "quantum": q}
