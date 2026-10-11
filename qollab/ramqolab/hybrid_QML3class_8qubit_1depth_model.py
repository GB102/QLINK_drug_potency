# NOTE: 'backend' is pre-created as a global variable
# (for local testing only:  from qiskit_aer import AerSimulator;  backend = AerSimulator())

import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.jobstatus import JobStatus

# ---------------- Files (adjust to where you uploaded them) ----------------
WEIGHTS_PATH = "hybrid_QML_3class_8qubit_1depth_basic_entangler.pt"
WEIGHTS_NPZ  = "hybrid_QML_3class_8qubit_1depth_basic_entangler.npz"   # fallback if torch is unavailable
TEST_CSV     = "pennylane_hybrid3_test_set_with_results.csv"           # the test set file saved earlier

# ---------------- Model settings ----------------
N_QUBITS, N_REPS = 8, 1
CLASS_NAMES = ["inactive", "intermediate", "active"]
BN_EPS = 1e-5            # PyTorch BatchNorm default
POLL_SECONDS = 10
USE_TRANSPILE = True    # set True if the backend rejects the circuit (for example, unsupported gates)


# ---------------- Load weights ----------------
def load_weights():
    try:
        import torch
        obj = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()}
    except ImportError:
        return {k: v.astype(np.float64) for k, v in np.load(WEIGHTS_NPZ).items()}

def get_circuit_weights(W):
    # PennyLane checkpoint: "qlayer.weights" (2, 4) | Qiskit checkpoint: "qlayer._weights" (8,), same layout
    key = "qlayer.weights" if "qlayer.weights" in W else "qlayer._weights"
    return W[key].reshape(N_REPS, N_QUBITS)

# ---------------- Classical part (NumPy, eval mode: dropout off, BN uses running stats) ----------------
def backbone_to_angles(X, W):
    h = X
    for k in (1, 2, 3):
        h = h @ W[f"fc{k}.weight"].T + W[f"fc{k}.bias"]
        h = (h - W[f"bn{k}.running_mean"]) / np.sqrt(W[f"bn{k}.running_var"] + BN_EPS) \
            * W[f"bn{k}.weight"] + W[f"bn{k}.bias"]
        h = np.maximum(h, 0.0)                                           # ReLU
    return np.tanh(h @ W["pre_net.weight"].T + W["pre_net.bias"]) * (np.pi / 2.0)   # (n, 4) angles

def head_probs(ev, W):
    logits = ev @ W["post_net.weight"].T + W["post_net.bias"]            # (n, 3)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

# ---------------- Quantum part: AngleEmbedding (RX) + BasicEntanglerLayers (2 reps) ----------------
def build_circuit(angles, w, measure=True):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS) if measure else QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):                      # angle embedding
        qc.rx(float(angles[i]), i)
    for r in range(N_REPS):                        # basic entangler layers
        for i in range(N_QUBITS):
            qc.rx(float(w[r, i]), i)
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)           # CNOT ring
    if measure:
        qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc

def expvals_from_counts(counts, n=N_QUBITS):
    """<Z_i> for each qubit from a counts / probabilities dict (Qiskit: rightmost bit = qubit 0)."""
    total = float(sum(counts.values()))
    ev = np.zeros(n)
    for bits, c in counts.items():
        if isinstance(bits, int):
            bits = format(bits, f"0{n}b")
        bits = bits.replace(" ", "")
        for i in range(n):
            ev[i] += c * (1.0 if bits[n - 1 - i] == "0" else -1.0)
    return ev / total

# ---------------- Job handling (same polling pattern as the template) ----------------
def wait_for(job):
    while True:
        status = job.status()
        print(f"Job status is {status}")
        match status:
            case JobStatus.DONE:
                return job.result()
            case JobStatus.ERROR:
                raise Exception(job.result())
        time.sleep(POLL_SECONDS)

def run_circuits(circuits, shots):
    if USE_TRANSPILE:
        from qiskit import transpile
        circuits = transpile(circuits, backend)
    try:
        job = backend.run(circuits, shots=shots)             # one job for all circuits
    except Exception as e:
        print(f"Batch submission not accepted ({e}); submitting one job per circuit")
        out = []
        for c in circuits:
            out.append(wait_for(backend.run(c, shots=shots)).get_counts())
        return out
    counts = wait_for(job).get_counts()
    return [counts] if isinstance(counts, dict) else list(counts)


# ---------------- Load everything once ----------------
W = load_weights()
W_circ = get_circuit_weights(W)

df = pd.read_csv(TEST_CSV)
feat_cols = [c for c in df.columns if c.startswith("PubchemFP") or (c.startswith("FP") and c[2:].isdigit())]
assert len(feat_cols) == W["fc1.weight"].shape[1], \
    f"{len(feat_cols)} feature columns in the CSV vs {W['fc1.weight'].shape[1]} expected by the model"
print(f"Loaded weights | test set: {len(df)} molecules, {len(feat_cols)} features")

# Example circuit (first test molecule), shown the way the template does
_angles0 = backbone_to_angles(df[feat_cols].values[:1].astype(np.float64), W)[0]
circuit = build_circuit(_angles0, W_circ)
print(circuit)


def main(
   backend,
    shots: int = 4096,
    excludeLowProbabilityValues: bool = True,
    lowProbabilityThreshold: float = 0.05
   
):
    # Use entire test set
    X = df[feat_cols].values.astype(np.float64)

    if "true_label" not in df.columns:
        raise ValueError(
            "true_label column not found in CSV. Cannot compute accuracy."
        )

    y_true = df["true_label"].values.astype(int)

    # Classical backbone
    angles = backbone_to_angles(X, W)

    # Build circuits for all molecules
    circuits = [
        build_circuit(a, W_circ)
        for a in angles
    ]

    # Submit job
    job = backend.run(circuits, shots=shots)

    # Poll for completion
    while True:
        status = job.status()
        print(f"Job status is {status}")

        match status:
            case JobStatus.DONE:
                break
            case JobStatus.ERROR:
                raise Exception(job.result())

        time.sleep(10)

    # Get counts for all circuits
    counts_list = job.result().get_counts()

    if isinstance(counts_list, dict):
        counts_list = [counts_list]

    # Optional display for first molecule
    shown_counts = counts_list[0]

    if excludeLowProbabilityValues:
        shotsThreshold = shots * lowProbabilityThreshold
        shown_counts = {
            bits: cnt
            for bits, cnt in shown_counts.items()
            if cnt > shotsThreshold
        }

    print(f"Counts for first molecule ({shots} shots): {shown_counts}")

    # Convert counts -> <Z> expectation values
    ev_hw = np.array([
        expvals_from_counts(c)
        for c in counts_list
    ])

    # Classical head
    prob_hw = head_probs(ev_hw, W)

    # Predictions
    pred_hw = prob_hw.argmax(axis=1)

    # Accuracy
    accuracy = (pred_hw == y_true).mean()

    result = {
      "model_name": WEIGHTS_NPZ,
      "total": len(y_true),
      "correct": (pred_hw == y_true).sum(),
      "incorrect": (pred_hw != y_true).sum(),
      "accuracy": accuracy,
      "accuracy_pct": accuracy * 100
    }
  
    return result

