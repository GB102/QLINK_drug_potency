"""Export a trained checkpoint into the Qollab project folder as plain JSON (no torch at demo time).

The Qollab runtime is Pyodide: Qiskit and NumPy are available, PyTorch is not. So the demo carries the
trained weights as JSON and rebuilds the circuit with Qiskit, then runs it on the selected backend.

Run:  python -m qlink_screen.export_qollab --run potent__qlink__d4__s0
"""
import argparse
import json

import numpy as np
import torch

from . import DATA_V2, ROOT
from .circuits import build
from .models import ANGLE_SCALE

QOLLAB = ROOT / "qollab"
N_DEMO = 60


def export(run_id, n_demo=N_DEMO):
    ck = torch.load(ROOT / "results" / "runs" / f"{run_id}.pt", map_location="cpu", weights_only=False)
    sd, run = ck["state_dict"], ck["run"]
    qc, xs, ws, obs = build(run["model"], run["n_blocks"])

    d = np.load(DATA_V2 / "features.npz", allow_pickle=True)
    import pandas as pd
    cur = pd.read_csv(DATA_V2 / "curated.csv")
    split = d["split"].astype(str)
    te = split == "test"
    X, pot, pic50 = d["compact"][te], d["potent"][te].astype(int), d["pIC50"][te]
    ids = cur.loc[te, "molecule_chembl_id"].tolist()
    smiles = cur.loc[te, "smiles"].tolist()

    # Deterministic demo pool: every n-th compound of the test split, preserving its potent prevalence.
    step = max(1, len(X) // n_demo)
    pick = list(range(0, len(X), step))[:n_demo]

    model = {
        "run_id": run_id, "model": run["model"], "n_blocks": run["n_blocks"], "seed": run["seed"],
        "task": run["task"], "n_qubits": qc.num_qubits, "n_data_qubits": len(obs),
        "angle_scale": ANGLE_SCALE,
        "pre_w": sd["pre.weight"].tolist(), "pre_b": sd["pre.bias"].tolist(),
        "circuit_weights": sd["weights"].tolist(),
        "post_w": sd["post.weight"].tolist(), "post_b": sd["post.bias"].tolist(),
        "val_metric": {run["selection_metric"]: run["best_val"]},
        "resources": run.get("resources", {}),
    }
    pool = {
        "target": "HIV-1 protease (ChEMBL CHEMBL243), enzymatic IC50",
        "label": "potent = median IC50 <= 10 nM (pIC50 >= 8)",
        "split": "held-out scaffold test split, seed 2026",
        "compounds": [{"id": ids[i], "smiles": smiles[i], "features": [round(float(v), 6) for v in X[i]],
                       "potent": int(pot[i]), "pIC50": round(float(pic50[i]), 3)} for i in pick],
    }
    QOLLAB.mkdir(exist_ok=True)
    (QOLLAB / "model_qlink.json").write_text(json.dumps(model))
    (QOLLAB / "pool.json").write_text(json.dumps(pool))
    print(f"exported {run_id}: {len(pick)} compounds, {sum(c['potent'] for c in pool['compounds'])} potent, "
          f"{qc.num_qubits} qubits, {len(sd['weights'])} circuit weights")
    return model, pool


def export_baseline(run_id="potent__rf_fp__d0__s0"):
    """Classical comparison scores for the demo pool, precomputed (the demo has no sklearn)."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    run = json.loads((ROOT / "results" / "runs" / f"{run_id}.json").read_text())
    d = np.load(DATA_V2 / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    X = np.hstack([d["ecfp4"], d["pubchem"]]).astype(np.float32)
    y = d["potent"].astype(int)
    m = RandomForestClassifier(**run["config"], class_weight="balanced", n_jobs=-1,
                               random_state=run["seed"]).fit(X[split == "train"], y[split == "train"])
    cur = pd.read_csv(DATA_V2 / "curated.csv")
    pool = json.loads((QOLLAB / "pool.json").read_text())
    want = {c["id"] for c in pool["compounds"]}
    mask = (split == "test") & cur.molecule_chembl_id.isin(want).values
    scores = dict(zip(cur.loc[mask, "molecule_chembl_id"], m.predict_proba(X[mask])[:, 1]))
    (QOLLAB / "classical_scores.json").write_text(json.dumps(
        {"model": "rf_fp (random forest, full fingerprints)", "trained_on": "train split only",
         "config": run["config"], "scores": {k: round(float(v), 6) for k, v in scores.items()}}))
    print(f"exported classical scores for {len(scores)} compounds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="potent__qlink__d4__s0")
    ap.add_argument("--n-demo", type=int, default=N_DEMO)
    a = ap.parse_args()
    export(a.run, a.n_demo)
    export_baseline()
