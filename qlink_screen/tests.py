"""Correctness checks for the circuits and the differentiable engine.

Run:  python -m qlink_screen.tests
Every check is a hard assert; the script prints a line per check and exits non-zero on failure.
"""
import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXXGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, Statevector

from . import circuits
from .quantum import CircuitLayer

RNG = np.random.default_rng(7)
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _bind(qc, xs, ws, x, w):
    return qc.assign_parameters({**{p: float(v) for p, v in zip(xs, x)},
                                 **{p: float(v) for p, v in zip(ws, w)}})


@check("Qiskit RXX == exp(-i t/2 X(x)X), matching TensorCircuit's rxx")
def t_rxx_convention():
    t = 0.7351
    xx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
    import scipy.linalg as sla
    assert np.allclose(Operator(RXXGate(t)).data, sla.expm(-1j * t / 2 * xx), atol=1e-12)


@check("torch engine reproduces Qiskit Statevector amplitudes for every model")
def t_statevector():
    for mid in circuits.SPECS:
        qc, xs, ws, obs = circuits.build(mid, n_blocks=3)
        x, w = RNG.normal(size=len(xs)), RNG.normal(size=len(ws))
        layer = CirciutLayerFor(qc, xs, ws, len(obs))
        ref = Statevector(_bind(qc, xs, ws, x, w)).data
        got = layer._state(torch.tensor(x, dtype=torch.float32)[None, :],
                           torch.tensor(w, dtype=torch.float32))[0].numpy()
        assert np.allclose(got, ref, atol=2e-5), (mid, np.abs(got - ref).max())


@check("torch engine reproduces Qiskit StatevectorEstimator <Z> for every model")
def t_expectations():
    est = StatevectorEstimator()
    for mid in circuits.SPECS:
        qc, xs, ws, obs = circuits.build(mid, n_blocks=4)
        x, w = RNG.normal(size=len(xs)), RNG.normal(size=len(ws))
        bound = _bind(qc, xs, ws, x, w)
        ref = np.array([est.run([(bound, o)]).result()[0].data.evs for o in obs], dtype=float)
        layer = CircuitLayer(qc, xs, ws, len(obs))
        got = layer(torch.tensor(x, dtype=torch.float32)[None, :],
                    torch.tensor(w, dtype=torch.float32))[0].numpy()
        assert np.allclose(got, ref, atol=2e-5), (mid, np.abs(got - ref).max())


@check("batched execution equals per-sample execution")
def t_batch():
    qc, xs, ws, obs = circuits.build("qlink", n_blocks=4)
    layer = CircuitLayer(qc, xs, ws, len(obs))
    x = torch.tensor(RNG.normal(size=(5, len(xs))), dtype=torch.float32)
    w = torch.tensor(RNG.normal(size=len(ws)), dtype=torch.float32)
    batch = layer(x, w)
    for i in range(5):
        assert torch.allclose(batch[i], layer(x[i:i + 1], w)[0], atol=1e-6)


@check("outputs depend on the inputs and on the trainable weights")
def t_dependence():
    for mid in circuits.SPECS:
        qc, xs, ws, obs = circuits.build(mid, n_blocks=4)
        layer = CircuitLayer(qc, xs, ws, len(obs))
        x0 = torch.tensor(RNG.normal(size=(1, len(xs))), dtype=torch.float32)
        w0 = torch.tensor(RNG.normal(size=len(ws)), dtype=torch.float32)
        base = layer(x0, w0)
        assert (layer(x0 + 0.4, w0) - base).abs().max() > 1e-3, f"{mid} ignores inputs"
        assert (layer(x0, w0 + 0.4) - base).abs().max() > 1e-3, f"{mid} ignores weights"


@check("autograd gradients match central finite differences (weights and inputs)")
def t_gradients():
    for mid in ["van4", "qlink", "qlink_sparse", "qlink_adaptive"]:
        qc, xs, ws, obs = circuits.build(mid, n_blocks=3)
        layer = CircuitLayer(qc, xs, ws, len(obs))
        x = torch.tensor(RNG.normal(size=(2, len(xs))), dtype=torch.float64, requires_grad=True)
        w = torch.tensor(RNG.normal(size=len(ws)), dtype=torch.float64, requires_grad=True)
        head = torch.tensor(RNG.normal(size=len(obs)), dtype=torch.float64)

        def loss(xx, ww):
            return (layer(xx.float(), ww.float()).double() @ head).sum()

        loss(x, w).backward()
        eps = 1e-3
        for t, g in ((w, w.grad), (x, x.grad)):
            flat = t.detach().reshape(-1)
            for j in RNG.choice(flat.numel(), size=min(4, flat.numel()), replace=False):
                p = flat.clone(); p[j] += eps
                m = flat.clone(); m[j] -= eps
                fd = (loss(x.detach(), p.reshape(t.shape)) if t is w else loss(p.reshape(t.shape), w.detach())) - \
                     (loss(x.detach(), m.reshape(t.shape)) if t is w else loss(m.reshape(t.shape), w.detach()))
                fd = float(fd) / (2 * eps)
                an = float(g.reshape(-1)[j])
                assert abs(fd - an) < 2e-2 * max(1.0, abs(fd)), (mid, j, fd, an)


@check("Q-LINK structure: messenger coherent, never measured/reset, not in observables")
def t_structure():
    for every in (1, 2):
        qc, xs, ws, obs, _ = circuits.qlink(n_data=4, n_blocks=4, every=every)
        assert qc.num_qubits == 5 and len(obs) == 4
        assert all(str(o.paulis[0])[0] == "I" for o in obs), "messenger must not be measured"
        names = [i.operation.name for i in qc.data]
        assert "measure" not in names and "reset" not in names
        n_rxx = sum(1 for i in qc.data if i.operation.name == "rxx")
        assert n_rxx == 4 * len(range(0, 4, every))


@check("sparse messenger halves the messenger gates and keeps the operation blocks")
def t_sparse():
    full, _, w_f, _, _ = circuits.qlink(n_data=4, n_blocks=8, every=1)
    sp, _, w_s, _, _ = circuits.qlink(n_data=4, n_blocks=8, every=2)
    assert len(w_f) == len(w_s), "same number of trainable block parameters"
    cnt = lambda c, g: sum(1 for i in c.data if i.operation.name == g)
    assert cnt(sp, "rxx") * 2 == cnt(full, "rxx") and cnt(sp, "cx") * 2 == cnt(full, "cx")
    assert cnt(sp, "cz") == cnt(full, "cz") == 8 * 3


@check("fixed coupling is pi/4 and the 5-qubit control leaves its extra qubit idle")
def t_controls():
    qc, _, _, _, _ = circuits.qlink(n_data=4, n_blocks=2, every=1)
    assert all(abs(float(i.operation.params[0]) - np.pi / 4) < 1e-12
               for i in qc.data if i.operation.name == "rxx")
    v5, xs, ws, obs = circuits.vanilla(n_data=4, n_blocks=4, n_idle=1)
    assert v5.num_qubits == 5 and len(xs) == 4
    touched = {q for i in v5.data for q in i.qubits}
    assert v5.qubits[4] not in touched, "the control's extra qubit must stay |0> and unused"
    v4, _, ws4, _ = circuits.vanilla(n_data=4, n_blocks=4)
    assert len(ws) == len(ws4), "5-qubit control must match the 4-qubit parameter count"


def CirciutLayerFor(qc, xs, ws, n_data):
    """CircuitLayer with a _state helper, used only by the amplitude test."""
    layer = CircuitLayer(qc, xs, ws, n_data)

    def _state(x, w):
        import torch as T
        from .quantum import CDTYPE, _PARAM_GATES, _apply
        b = x.shape[0]
        st = T.zeros(b, 2 ** layer.n, dtype=CDTYPE)
        st[:, 0] = 1.0
        for name, kind, idx, axes in layer.ops:
            if kind is None:
                mat = layer._const_mat(name, x.device)
            elif kind == "x":
                mat = _PARAM_GATES[name](x[:, idx])
            elif kind == "w":
                mat = _PARAM_GATES[name](w[idx])
            else:
                mat = _PARAM_GATES[name](T.tensor(idx))
            st = _apply(st, mat, axes)
        return st

    layer._state = _state
    return layer


if __name__ == "__main__":
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    raise SystemExit(1 if failed else 0)
