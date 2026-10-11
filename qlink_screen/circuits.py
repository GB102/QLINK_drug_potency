"""Qiskit circuit builders: team's BasicEntangler family, the Q-LINK paper's vanilla block, and Q-LINK.

Q-LINK method from arXiv:2604.11831 (Q-LINK: a messenger qubit for residual-like information flow).
Gate order follows the authors' reference implementation
(github.com/AARC-lab/DCAS_2026_QLINK, src/qlink/circuit/quantum_circuit.py, TensorCircuit):

    messenger: H
    RXX(theta) between every data qubit i and the messenger          <- collection
    for each block d:
        RZ, RY, RX on every data qubit (trainable)
        CZ chain between neighbouring data qubits
        CNOT(messenger -> i) for every data qubit i                  <- distribution
        if d is not the last block: RXX(theta) collection for the next block

Fixed variant: theta = pi/4 for every coupling. Adaptive variant: theta trainable per (block, qubit).
RXX convention: Qiskit's RXXGate is exp(-i*theta/2 * X⊗X); TensorCircuit's rxx is the same
exp(-i*theta/2 XX), so theta=pi/4 carries over unchanged (verified in tests.py).

Our transfer to classification differs from the paper in two declared ways: the input state is not a random
quantum state but an RX angle encoding of 4 classical features, and the objective is supervised
cross-entropy on <Z> read-outs instead of the paper's ground-state cost. The messenger is never measured
or reset; it stays coherent to the end of the circuit and is excluded from the observables.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

FIXED_COUPLING = np.pi / 4


def _observables(n_data, n_total):
    """<Z_i> on each data qubit. Qiskit is little-endian: rightmost label character is qubit 0."""
    return [SparsePauliOp("I" * (n_total - 1 - i) + "Z" + "I" * i) for i in range(n_data)]


def basic_entangler(n_qubits=4, n_blocks=4):
    """The team's original circuit: RX angle embedding, then n_blocks x [RX all qubits, CNOT ring]."""
    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", n_blocks * n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(x[i], i)
    for d in range(n_blocks):
        for i in range(n_qubits):
            qc.rx(w[d * n_qubits + i], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    return qc, list(x), list(w), _observables(n_qubits, n_qubits)


def _operation_block(qc, w, d, n_data):
    """Paper Eq. (2) block, in the reference implementation's order: RZ, RY, RX, then a CZ chain."""
    for i in range(n_data):
        qc.rz(w[d][i][0], i)
        qc.ry(w[d][i][1], i)
        qc.rx(w[d][i][2], i)
    for i in range(n_data - 1):
        qc.cz(i, i + 1)


def vanilla(n_data=4, n_blocks=4, n_idle=0):
    """Ordinary VQC with the paper's operation block. n_idle adds unused |0> qubits (the 5-qubit control)."""
    n_total = n_data + n_idle
    x = ParameterVector("x", n_data)
    wv = ParameterVector("w", n_blocks * n_data * 3)
    w = np.array(list(wv), dtype=object).reshape(n_blocks, n_data, 3)
    qc = QuantumCircuit(n_total)
    for i in range(n_data):
        qc.rx(x[i], i)
    for d in range(n_blocks):
        _operation_block(qc, w, d, n_data)
    return qc, list(x), list(wv), _observables(n_data, n_total)


def qlink(n_data=4, n_blocks=4, adaptive=False, every=1):
    """Q-LINK with one messenger qubit (the highest-index wire).

    every=1 couples the messenger around every block (the paper's method);
    every=2 is our sparse-messenger variant: collection/distribution only around blocks 0, 2, 4, ...
    """
    n_total = n_data + 1
    msg = n_data
    x = ParameterVector("x", n_data)
    wv = ParameterVector("w", n_blocks * n_data * 3)
    w = np.array(list(wv), dtype=object).reshape(n_blocks, n_data, 3)
    coupled = [d for d in range(n_blocks) if d % every == 0]
    res = ParameterVector("r", len(coupled) * n_data) if adaptive else None
    theta = {}
    for k, d in enumerate(coupled):
        for i in range(n_data):
            theta[(d, i)] = res[k * n_data + i] if adaptive else FIXED_COUPLING

    qc = QuantumCircuit(n_total)
    for i in range(n_data):
        qc.rx(x[i], i)
    qc.h(msg)                                   # reference impl: H on the messenger before the first coupling
    for d in range(n_blocks):
        if d in coupled:                        # collection
            for i in range(n_data):
                qc.rxx(theta[(d, i)], i, msg)
        _operation_block(qc, w, d, n_data)
        if d in coupled:                        # distribution
            for i in range(n_data):
                qc.cx(msg, i)
    return qc, list(x), list(wv), _observables(n_data, n_total), (list(res) if adaptive else [])


SPECS = {
    # id: (builder, kwargs) -- n_blocks is supplied at build time
    "be4": (basic_entangler, {"n_qubits": 4}),
    "van4": (vanilla, {"n_data": 4, "n_idle": 0}),
    "van5": (vanilla, {"n_data": 4, "n_idle": 1}),
    "qlink": (qlink, {"n_data": 4, "every": 1}),
    "qlink_sparse": (qlink, {"n_data": 4, "every": 2}),
    "qlink_adaptive": (qlink, {"n_data": 4, "every": 1, "adaptive": True}),
}


def build(model_id, n_blocks):
    """Return (circuit, input_params, weight_params, observables). Adaptive couplings join weight_params."""
    fn, kw = SPECS[model_id]
    out = fn(n_blocks=n_blocks, **kw)
    if len(out) == 5:
        qc, xs, ws, obs, res = out
        return qc, xs, ws + res, obs
    return out
