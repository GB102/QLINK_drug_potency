"""Q-LINK circuit builder for the Qollab runtime (Qiskit only, no torch).

Identical gate sequence to qlink_screen/circuits.py, which is the version used for training and which is
checked against Qiskit's own Statevector/Estimator in qlink_screen/tests.py.

Q-LINK method: arXiv:2604.11831. Reference implementation: github.com/AARC-lab/DCAS_2026_QLINK.
"""
import math

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

FIXED_COUPLING = math.pi / 4


def qlink_circuit(n_data=4, n_blocks=4, every=1, messenger=True):
    """Returns (circuit, input_params, weight_params). Messenger is the highest-index qubit."""
    n_total = n_data + (1 if messenger else 0)
    x = ParameterVector("x", n_data)
    w = ParameterVector("w", n_blocks * n_data * 3)
    qc = QuantumCircuit(n_total)
    for i in range(n_data):
        qc.rx(x[i], i)
    if messenger:
        msg = n_data
        qc.h(msg)
    coupled = set(range(0, n_blocks, every))
    for d in range(n_blocks):
        if messenger and d in coupled:                      # collection
            for i in range(n_data):
                qc.rxx(FIXED_COUPLING, i, msg)
        for i in range(n_data):                             # operation block: RZ, RY, RX
            base = (d * n_data + i) * 3
            qc.rz(w[base], i)
            qc.ry(w[base + 1], i)
            qc.rx(w[base + 2], i)
        for i in range(n_data - 1):                         # CZ chain
            qc.cz(i, i + 1)
        if messenger and d in coupled:                      # distribution
            for i in range(n_data):
                qc.cx(msg, i)
    return qc, list(x), list(w)


def build_from_spec(spec):
    """spec: the exported model_qlink.json dict."""
    model = spec["model"]
    n_data = spec["n_data_qubits"]
    kw = {"qlink": dict(every=1, messenger=True),
          "qlink_sparse": dict(every=2, messenger=True),
          "van4": dict(every=1, messenger=False),
          "van5": dict(every=1, messenger=False)}[model]
    return qlink_circuit(n_data=n_data, n_blocks=spec["n_blocks"], **kw)


def encode_angles(features, pre_w, pre_b, angle_scale):
    """Linear(16->4) then (pi/2)*tanh, matching the trained model's encoder."""
    return [angle_scale * math.tanh(sum(a * b for a, b in zip(row, features)) + bias)
            for row, bias in zip(pre_w, pre_b)]


def head(z_values, post_w, post_b):
    """Linear(4->1) then logistic, matching the trained model's prediction head."""
    logit = sum(a * b for a, b in zip(post_w[0], z_values)) + post_b[0]
    return 1.0 / (1.0 + math.exp(-logit)), logit


def packed_circuit(qc, xs, ws, angle_rows, weights, n_data):
    """One circuit holding several compounds side by side, each on its own block of qubits.

    IonQ accepts a single circuit per job, so a run that measures N compounds has to put all N in one
    circuit. The blocks share no gates, so the N sub-circuits stay independent and each shot yields an
    independent sample for every compound; per-compound statistics come from marginalising the bitstring.
    Compound c occupies qubits [c*nq, (c+1)*nq) and classical bits [c*n_data, (c+1)*n_data).
    """
    nq = qc.num_qubits
    n = len(angle_rows)
    out = QuantumCircuit(nq * n, n_data * n)
    for c, angles in enumerate(angle_rows):
        bound = qc.assign_parameters({**{p: float(v) for p, v in zip(xs, angles)},
                                      **{p: float(v) for p, v in zip(ws, weights)}})
        base = c * nq
        out.compose(bound, qubits=range(base, base + nq), inplace=True)
        out.measure(range(base, base + n_data), range(c * n_data, c * n_data + n_data))
    return out


def z_from_counts(counts, n_data, offset=0):
    """<Z_i> for data qubits from measurement counts. Qiskit bitstrings are little-endian.

    `offset` selects one compound's block of classical bits inside a packed circuit.
    """
    total = sum(counts.values())
    z = [0.0] * n_data
    for bits, c in counts.items():
        bits = bits.replace(" ", "")
        for i in range(n_data):
            bit = int(bits[len(bits) - 1 - (offset + i)])
            z[i] += c * (1 - 2 * bit)
    return [v / total for v in z], total
