# NOTE: 'backend' is pre-created as a global variable
# (for local testing only:  from qiskit_aer import AerSimulator;  backend = AerSimulator())

import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ---------------- Models to evaluate (edit the file names if yours differ) ----------------
TEST_CSV = "pennylane_hybrid3_test_set_with_results.csv"     # test set saved earlier (same for all models)

# Every weight file is named as a PLAIN STRING LITERAL. Qollab only loads a project data file when its
# path appears literally in the source: a name built at run time (an f-string, os.path.join, +) is not
# picked up and open() then fails with OSError errno 29. Do not refactor these into a helper.
MODELS = [
    {"name": "4qubit_2depth",  "weights": "hybrid_QML_3class_4qubit_2depth_basic_entangler.npz"},
    {"name": "4qubit_4depth",  "weights": "hybrid_QML_3class_4qubit_4depth_basic_entangler.npz"},
    {"name": "4qubit_8depth",  "weights": "hybrid_QML_3class_4qubit_8depth_basic_entangler.npz"},
    {"name": "8qubit_1depth",  "weights": "hybrid_QML_3class_8qubit_1depth_basic_entangler.npz"},
    {"name": "8qubit_2depth",  "weights": "hybrid_QML_3class_8qubit_2depth_basic_entangler.npz"},
    {"name": "8qubit_4depth",  "weights": "hybrid_QML_3class_8qubit_4depth_basic_entangler.npz"},
    {"name": "16qubit_1depth", "weights": "hybrid_QML_3class_16qubit_1depth_basic_entangler.npz"},
]

CLASS_NAMES = ["inactive", "intermediate", "active"]
BN_EPS = 1e-5            # PyTorch BatchNorm default

# ---------------- Run settings (one job per code run; each run executes ONE part of the plan) ----------------
BACKEND_BATCH = 1        # molecules per model in this run's single circuit
START_INDEX   = 0        # first test molecule of the backend slice
PART          = 0        # which group of models runs on the backend in this run (0, 1, ...); the plan is printed
QUBIT_LIMIT   = 29       # used only if the backend does not report its qubit count
HEADLINE      = "exact"  # "exact": returned accuracy = whole test set, exact expectations (local)
                         # "backend": returned accuracy = this run's backend slice
EXACT_MAX     = None     # None = exact accuracy on the whole test set; an int limits it (faster)


# ---------------- Loading ----------------
_CSV_CACHE = {}

def read_csv(path):
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]

def load_state(path):
    """Weights as float64 NumPy arrays, from a .npz (NumPy) or a .pt (needs torch) file."""
    if path.endswith(".npz"):
        return {k: v.astype(np.float64) for k, v in np.load(path).items()}
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                  # very old torch without weights_only
        obj = torch.load(path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()}


def circuit_weights(W):
    """(reps, n_qubits) circuit weights from a PennyLane ('qlayer.weights') or Qiskit ('qlayer._weights') file."""
    key = "qlayer.weights" if "qlayer.weights" in W else "qlayer._weights"
    nq = W["pre_net.weight"].shape[0]
    assert W["post_net.weight"].shape[1] == nq, "pre_net / post_net sizes disagree"
    flat = W[key].reshape(-1)
    assert flat.size % nq == 0, "circuit weight count is not a multiple of the qubit count"
    return flat.reshape(-1, nq)

# ---------------- Classical part (NumPy, eval mode: dropout off, BN uses running stats) ----------------
def backbone_to_angles(X, W):
    h = X
    for k in (1, 2, 3):
        h = h @ W[f"fc{k}.weight"].T + W[f"fc{k}.bias"]
        h = (h - W[f"bn{k}.running_mean"]) / np.sqrt(W[f"bn{k}.running_var"] + BN_EPS) \
            * W[f"bn{k}.weight"] + W[f"bn{k}.bias"]
        h = np.maximum(h, 0.0)                                           # ReLU
    return np.tanh(h @ W["pre_net.weight"].T + W["pre_net.bias"]) * (np.pi / 2.0)   # (n, n_qubits)

def head_probs(ev, W):
    logits = ev @ W["post_net.weight"].T + W["post_net.bias"]            # (n, 3)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

# ---------------- Quantum part: AngleEmbedding (RX) + BasicEntanglerLayers ----------------
def build_circuit(angles, w, measure=True):
    """One molecule: len(angles) qubits, w has shape (reps, n_qubits)."""
    nq = len(angles)
    qc = QuantumCircuit(nq, nq) if measure else QuantumCircuit(nq)
    for i in range(nq):                                  # angle embedding
        qc.rx(float(angles[i]), i)
    for r in range(w.shape[0]):                          # basic entangler layers
        for i in range(nq):
            qc.rx(float(w[r, i]), i)
        for i in range(nq):
            qc.cx(i, (i + 1) % nq)                       # CNOT ring
    if measure:
        qc.measure(range(nq), range(nq))
    return qc

def packed_circuit(blocks):
    """blocks: list of (angles, w). All blocks side by side in ONE circuit; block k uses its own range of
    qubits and classical bits. The blocks are not entangled, so each shot samples all of them at once."""
    total = sum(len(a) for a, _ in blocks)
    qc = QuantumCircuit(total, total)
    off = 0
    for angles, w in blocks:
        nq = len(angles)
        for i in range(nq):
            qc.rx(float(angles[i]), off + i)
        for r in range(w.shape[0]):
            for i in range(nq):
                qc.rx(float(w[r, i]), off + i)
            for i in range(nq):
                qc.cx(off + i, off + (i + 1) % nq)
        off += nq
    qc.measure(range(total), range(total))
    return qc

def z_from_counts(counts, offset, width, n):
    """<Z_i> for the n qubits starting at `offset`, from a counts dict whose bitstrings are `width` bits
    long (Qiskit: rightmost bit = qubit 0)."""
    total = float(sum(counts.values()))
    ev = np.zeros(n)
    for bits, c in counts.items():
        if isinstance(bits, int):
            bits = format(bits, f"0{width}b")
        bits = str(bits).replace(" ", "")
        if bits.startswith("0x"):                        # hex key -> binary string
            bits = format(int(bits, 16), f"0{width}b")
        bits = bits.zfill(width)
        for i in range(n):
            ev[i] += c * (1.0 if bits[-(1 + offset + i)] == "0" else -1.0)
    return ev / total

_SIGNS = {}
def _signs(nq):
    if nq not in _SIGNS:
        idx = np.arange(2 ** nq)
        _SIGNS[nq] = np.array([1.0 - 2.0 * ((idx >> i) & 1) for i in range(nq)])   # (nq, 2**nq)
    return _SIGNS[nq]

def exact_expvals(angles, w):
    """Noiseless <Z> per qubit for every molecule (local statevector, no backend call)."""
    nq = w.shape[1]
    s = _signs(nq)
    out = np.empty((len(angles), nq))
    for k, a in enumerate(angles):
        out[k] = s @ Statevector(build_circuit(a, w, measure=False)).probabilities()
    return out

def empty_result(spec):
    return {"model_name": spec["weights"], "n_qubits": None, "depth": None,
            "total": None, "correct": None, "incorrect": None, "accuracy": None, "accuracy_pct": None,
            "backend": None, "backend_shots": None, "backend_slice_start": None,
            "backend_slice_total": None, "backend_slice_correct": None, "backend_slice_accuracy": None,
            "status": ""}


def main(backend=None, shots: int = 1024, backend_batch=None, part=None, start_index=None):
    # Accept main(backend, shots=...), main(shots=...) and main(1024)
    if isinstance(backend, (int, float, str)):
        shots, backend = backend, None
    if backend is None:
        backend = globals().get("backend")
    if backend is None:
        raise RuntimeError("no backend available")

    shots = int(shots)
    assert shots > 0, f"bad shots value: {shots!r}"
    b_req = max(1, int(BACKEND_BATCH if backend_batch is None else backend_batch))
    part = int(PART if part is None else part)
    lo = int(START_INDEX if start_index is None else start_index)

    results = [empty_result(spec) for spec in MODELS]
    loaded = []

    # ======== Step 1. Load every model and compute exact accuracy locally ========
    print("=" * 78)
    print("Step 1  Exact (noiseless) accuracy per model, computed locally")
    print("=" * 78)
    for idx, spec in enumerate(MODELS):
        res = results[idx]
        try:
            W = load_state(spec["weights"])
            w = circuit_weights(W)
            df = read_csv(spec.get("csv", TEST_CSV))
            feat_cols = [c for c in df.columns
                         if c.startswith("PubchemFP") or (c.startswith("FP") and c[2:].isdigit())]
            if len(feat_cols) != W["fc1.weight"].shape[1]:
                raise ValueError(f"{len(feat_cols)} feature columns in the CSV vs "
                                 f"{W['fc1.weight'].shape[1]} expected by the model")
            if "true_label" not in df.columns:
                raise ValueError("true_label column not found in the CSV")

            y = df["true_label"].values.astype(int)
            angles = backbone_to_angles(df[feat_cols].values.astype(np.float64), W)
            if not (np.isfinite(angles).all() and np.isfinite(w).all()):
                raise ValueError("NaN/inf in circuit angles or weights")

            n_all = len(y)
            n_ex = n_all if EXACT_MAX is None else min(n_all, max(int(EXACT_MAX), lo + b_req))
            t0 = time.time()
            ev_exact = exact_expvals(angles[:n_ex], w)
            pred_exact = head_probs(ev_exact, W).argmax(axis=1)
            correct = int((pred_exact == y[:n_ex]).sum())

            m = {"idx": idx, "spec": spec, "W": W, "w": w, "nq": w.shape[1], "reps": w.shape[0],
                 "y": y, "ids": df["molecule_chembl_id"].values if "molecule_chembl_id" in df.columns
                                else np.arange(n_all),
                 "angles": angles, "ev_exact": ev_exact, "pred_exact": pred_exact, "res": res}
            loaded.append(m)

            res.update(n_qubits=int(m["nq"]), depth=int(m["reps"]), total=int(n_ex), correct=correct,
                       incorrect=int(n_ex - correct), accuracy=correct / n_ex,
                       accuracy_pct=100.0 * correct / n_ex, status="exact only (not on the backend in this run)")
            print(f"  {spec['name']:<16} {m['nq']:>2} qubits, {m['reps']} reps | exact accuracy "
                  f"{correct / n_ex:.4f} ({correct}/{n_ex}) | {time.time() - t0:.1f}s")
        except Exception as e:
            res["status"] = f"skipped: {type(e).__name__}: {e}"
            print(f"  {spec['name']:<16} SKIPPED ({type(e).__name__}: {e})")
    print()

    # ======== Step 2. Plan the backend run: ONE job, so pack models and molecules into one circuit ========
    print("=" * 78)
    print("Step 2  Backend plan (one job per code run)")
    print("=" * 78)
    name = getattr(backend, "name", None)
    name = name() if callable(name) else (name or type(backend).__name__)
    is_ionq = "ionq" in str(name).lower()
    ceiling = getattr(backend, "num_qubits", None)
    if not isinstance(ceiling, int) or ceiling < 1:
        ceiling = QUBIT_LIMIT

    cands = []
    for m in loaded:
        n_m = len(m["y"])
        if lo >= n_m:
            m["res"]["status"] = f"exact only: start_index {lo} is past the end of its test set"
            continue
        count = min(b_req, n_m - lo)
        if is_ionq:
            if m["nq"] > ceiling:
                m["res"]["status"] = f"exact only: needs {m['nq']} qubits, backend has {ceiling}"
                print(f"  {m['spec']['name']}: needs {m['nq']} qubits but the backend has {ceiling}; exact only")
                continue
            count = min(count, ceiling // m["nq"])
        m["lo"], m["hi"] = lo, lo + count
        m["cost"] = m["nq"] * count
        cands.append(m)

    if is_ionq:                                           # first-fit decreasing into groups of <= ceiling qubits
        bins = []
        for m in sorted(cands, key=lambda m: -m["cost"]):
            for b in bins:
                if b[0] + m["cost"] <= ceiling:
                    b[0] += m["cost"]
                    b[1].append(m)
                    break
            else:
                bins.append([m["cost"], [m]])
    else:                                                 # local simulators accept many circuits per job
        bins = [[sum(m["cost"] for m in cands), cands]] if cands else []

    print(f"  backend {name} | qubits available: {ceiling if is_ionq else 'n/a'} | shots {shots} | "
          f"molecules per model: up to {b_req} (from index {lo})")
    for p, (used, members) in enumerate(bins):
        members = sorted(members, key=lambda m: m["idx"])
        tag = "   <-- this run" if p == part else ""
        print(f"  part {p}: " + ", ".join(f"{m['spec']['name']} ({m['cost']}q)" for m in members)
              + f"  [{used} qubits]{tag}")
    print(f"  This file needs {len(bins)} run(s) to cover every model on the backend: PART = 0 .. {max(len(bins) - 1, 0)}")
    print()

    # ======== Step 3. Live backend run ========
    if part >= len(bins):
        print(f"  PART = {part} is past the last part ({len(bins) - 1}); nothing to run on the backend in this run.")
    else:
        chosen = sorted(bins[part][1], key=lambda m: m["idx"])
        blocks = [(m, j) for m in chosen for j in range(m["lo"], m["hi"])]
        print("=" * 78)
        print(f"Step 3  Live backend run (executed now): part {part}, {len(chosen)} models, {len(blocks)} molecule blocks")
        print("=" * 78)
        try:
            if is_ionq:
                payload = packed_circuit([(m["angles"][j], m["w"]) for m, j in blocks])
                print(f"  submission : 1 circuit of {payload.num_qubits} qubits, {sum(payload.count_ops().values())} "
                      f"operations, 1 job")
            else:
                payload = [build_circuit(m["angles"][j], m["w"]) for m, j in blocks]
                print(f"  submission : {len(payload)} circuits, 1 job")
            print("  running ...")
            job = backend.run(payload, shots=shots)
            counts = job.result().get_counts()

            ev_by_model = {m["idx"]: [] for m in chosen}
            if is_ionq:
                if isinstance(counts, list):
                    counts = counts[0]
                width, off = payload.num_qubits, 0
                for m, j in blocks:
                    ev_by_model[m["idx"]].append(z_from_counts(counts, off, width, m["nq"]))
                    off += m["nq"]
            else:
                counts_list = counts if isinstance(counts, list) else [counts]
                for (m, j), c in zip(blocks, counts_list):
                    ev_by_model[m["idx"]].append(z_from_counts(c, 0, m["nq"], m["nq"]))

            shot_sigma = 1.0 / np.sqrt(shots)
            print(f"\n  1-sigma shot noise on a single <Z> is about {shot_sigma:.3f}\n")
            print(f"  {'model':<16}{'backend acc':>13}{'exact acc':>11}{'agree':>8}{'mean|dZ|':>10}")
            for m in chosen:
                ev_hw = np.array(ev_by_model[m["idx"]])
                a, b = m["lo"], m["hi"]
                y_s = m["y"][a:b]
                pred_hw = head_probs(ev_hw, m["W"]).argmax(axis=1)
                ev_ex = m["ev_exact"][a:b]
                pred_ex = m["pred_exact"][a:b]
                c_hw = int((pred_hw == y_s).sum())
                n_s = b - a
                print(f"  {m['spec']['name']:<16}{f'{c_hw}/{n_s}':>13}{f'{int((pred_ex == y_s).sum())}/{n_s}':>11}"
                      f"{f'{int((pred_hw == pred_ex).sum())}/{n_s}':>8}{np.abs(ev_hw - ev_ex).mean():>10.4f}")
                if len(blocks) <= 20:
                    for k in range(n_s):
                        print(f"      #{a + k:<5}{str(m['ids'][a + k]):<16} true {CLASS_NAMES[y_s[k]]:<13}"
                              f"exact {CLASS_NAMES[pred_ex[k]]:<13}backend {CLASS_NAMES[pred_hw[k]]}")
                print(f"      RESULT model={m['spec']['name']} start={a} end={b} correct={c_hw} total={n_s}")

                r = m["res"]
                r.update(backend=str(name), backend_shots=shots, backend_slice_start=int(a),
                         backend_slice_total=int(n_s), backend_slice_correct=c_hw,
                         backend_slice_accuracy=c_hw / n_s, status=f"backend run in part {part}")
                if HEADLINE == "backend":
                    r.update(total=int(n_s), correct=c_hw, incorrect=int(n_s - c_hw),
                             accuracy=c_hw / n_s, accuracy_pct=100.0 * c_hw / n_s)
                if is_ionq and np.abs(ev_ex).mean() - np.abs(ev_hw).mean() > shot_sigma:
                    print(f"      note: |<Z>| is smaller on the backend than exact "
                          f"({np.abs(ev_hw).mean():.3f} vs {np.abs(ev_ex).mean():.3f}): consistent with a "
                          f"device noise model, not just finite sampling")
        except Exception as e:
            print(f"\n  Backend run failed ({type(e).__name__}: {e})")
            print("  The exact results above are still returned. Lower BACKEND_BATCH or choose a different PART, then re-run.")
            for m in chosen:
                m["res"]["status"] = f"exact only; backend run failed: {type(e).__name__}: {e}"

        print(f"\n  Next: run PART = {part + 1} to cover the next group of models" if part + 1 < len(bins)
              else "\n  All parts of this plan have been run (for this start index).")
        print(f"  For more molecules: set START_INDEX = {lo + b_req}.")

    if HEADLINE == "backend":                              # models not on the backend in this run have no backend figure
        for r in results:
            if r["backend_slice_total"] is None:
                r.update(total=None, correct=None, incorrect=None, accuracy=None, accuracy_pct=None)

    return results