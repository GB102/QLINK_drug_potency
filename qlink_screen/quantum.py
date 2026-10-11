"""Differentiable statevector execution of a Qiskit circuit.

The Qiskit `QuantumCircuit` objects in `circuits.py` are the single source of truth. This module compiles
one of them into a list of (matrix-builder, qubits) operations on a batched torch statevector, so training
differentiates the exact same circuit that the Qollab/IonQ demo submits. Exactness is checked against
`qiskit.quantum_info.Statevector` and `StatevectorEstimator` in `tests.py`.

Gradients come from torch autograd through the simulated amplitudes (exact, no parameter-shift sampling).
For circuit expectations this is numerically equivalent to the shift rule and far cheaper; the chain rule
into the classical head is handled by autograd, never by applying the shift rule to the loss itself.
"""
import numpy as np
import torch
from qiskit.circuit import ParameterExpression

CDTYPE = torch.complex64


def _rx(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2).to(CDTYPE) + \
        torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2).to(CDTYPE) * 1j


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CDTYPE)


def _rz(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2).to(CDTYPE) + \
        torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2).to(CDTYPE) * 1j


def _rxx(t):
    """exp(-i t/2 X(x)X) -- Qiskit's RXXGate convention, identical to TensorCircuit's rxx."""
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    re = torch.stack([torch.stack([c, z, z, z], -1), torch.stack([z, c, z, z], -1),
                      torch.stack([z, z, c, z], -1), torch.stack([z, z, z, c], -1)], -2)
    im = torch.stack([torch.stack([z, z, z, -s], -1), torch.stack([z, z, -s, z], -1),
                      torch.stack([z, -s, z, z], -1), torch.stack([-s, z, z, z], -1)], -2)
    return re.to(CDTYPE) + im.to(CDTYPE) * 1j


_PARAM_GATES = {"rx": _rx, "ry": _ry, "rz": _rz, "rxx": _rxx}
_CONST_GATES = {
    "h": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "x": np.array([[0, 1], [1, 0]]),
    "cz": np.diag([1, 1, 1, -1]),
    # little-endian: index = q1*2 + q0, control is the first circuit argument
    "cx": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
}


def _apply(state, mat, axes):
    """state (B, 2**n) complex; mat (2**k,2**k) or (B,2**k,2**k); axes = tensor axes, most significant first."""
    b = state.shape[0]
    n = state.shape[1].bit_length() - 1
    k = len(axes)
    t = state.reshape(b, *([2] * n))
    t = torch.movedim(t, axes, list(range(1, k + 1)))
    shape = t.shape
    t = torch.matmul(mat, t.reshape(b, 2 ** k, -1))
    return torch.movedim(t.reshape(shape), list(range(1, k + 1)), axes).reshape(b, -1)


class CircuitLayer(torch.nn.Module):
    """Executes a Qiskit circuit on a batched statevector and returns <Z> on the first `n_data` qubits."""

    def __init__(self, circuit, input_params, weight_params, n_data):
        super().__init__()
        self.n = circuit.num_qubits
        self.n_data = n_data
        self.n_weights = len(weight_params)
        slot = {p: ("x", i) for i, p in enumerate(input_params)}
        slot.update({p: ("w", i) for i, p in enumerate(weight_params)})
        qpos = {q: i for i, q in enumerate(circuit.qubits)}
        self.ops = []
        for inst in circuit.data:
            name = inst.operation.name
            if name == "barrier":
                continue
            # most significant gate-basis bit = last circuit argument (Qiskit little-endian)
            axes = [self.n - qpos[q] for q in reversed(inst.qubits)]
            if name in _CONST_GATES:
                self.ops.append((name, None, None, axes))
                continue
            if name not in _PARAM_GATES:
                raise ValueError(f"unsupported gate for the torch engine: {name}")
            (p,) = inst.operation.params
            if isinstance(p, ParameterExpression) and p.parameters:
                kind, idx = slot[p]
                self.ops.append((name, kind, idx, axes))
            else:
                self.ops.append((name, "const", float(p), axes))
        self.register_buffer("_const", torch.tensor(0.0))
        self._cache = {}
        signs = np.stack([1 - 2 * ((np.arange(2 ** self.n) >> q) & 1) for q in range(n_data)])
        self.register_buffer("z_signs", torch.tensor(signs, dtype=torch.float32))

    def _const_mat(self, name, device):
        key = (name, device)
        if key not in self._cache:
            self._cache[key] = torch.tensor(_CONST_GATES[name], dtype=CDTYPE, device=device)
        return self._cache[key]

    def forward(self, x, weights):
        """x: (B, n_inputs) encoding angles; weights: (n_weights,). Returns (B, n_data) in [-1, 1]."""
        b, device = x.shape[0], x.device
        state = torch.zeros(b, 2 ** self.n, dtype=CDTYPE, device=device)
        state[:, 0] = 1.0
        for name, kind, idx, axes in self.ops:
            if kind is None:
                mat = self._const_mat(name, device)
            elif kind == "x":
                mat = _PARAM_GATES[name](x[:, idx])
            elif kind == "w":
                mat = _PARAM_GATES[name](weights[idx])
            else:
                mat = _PARAM_GATES[name](torch.tensor(idx, dtype=torch.float32, device=device))
            state = _apply(state, mat, axes)
        probs = state.real ** 2 + state.imag ** 2
        return probs @ self.z_signs.T
