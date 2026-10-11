"""Compiled resource accounting: depth and two-qubit gate counts for a generic basis and IonQ's native set.

IonQ's native two-qubit gate is the Mølmer-Sørensen family (zz / ms); a published circuit is compiled into
that set before it runs, which changes the gate count. Both are reported so the messenger's real overhead
is visible rather than asserted from the circuit diagram.
"""
from qiskit import transpile

GENERIC_BASIS = ["rz", "ry", "rx", "cx"]
IONQ_BASIS = ["rz", "ry", "rx", "rzz"]   # rzz stands in for the MS/ZZ two-qubit interaction
TWO_Q = {"cx", "cz", "rxx", "ryy", "rzz", "ms", "zz", "swap", "ecr"}


def _counts(qc):
    ops = qc.count_ops()
    return {"depth": qc.depth(), "n_qubits": qc.num_qubits, "gates": int(sum(ops.values())),
            "two_qubit_gates": int(sum(v for k, v in ops.items() if k in TWO_Q)),
            "ops": {k: int(v) for k, v in ops.items() if k != "barrier"}}


def circuit_resources(qc, seed=11):
    """Logical counts plus compiled counts in both bases (all-to-all coupling, as on IonQ)."""
    out = {"logical": _counts(qc)}
    for name, basis in (("generic_cx", GENERIC_BASIS), ("ionq_native", IONQ_BASIS)):
        t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=seed)
        out[name] = _counts(t)
    return out


if __name__ == "__main__":
    import json

    from . import circuits
    table = {}
    for mid in circuits.SPECS:
        for d in (4, 8):
            qc, _, ws, _ = circuits.build(mid, d)
            r = circuit_resources(qc)
            table[f"{mid}_d{d}"] = {"trainable_circuit_params": len(ws),
                                    "qubits": qc.num_qubits,
                                    "logical_2q": r["logical"]["two_qubit_gates"],
                                    "ionq_2q": r["ionq_native"]["two_qubit_gates"],
                                    "ionq_depth": r["ionq_native"]["depth"]}
    print(json.dumps(table, indent=2))
