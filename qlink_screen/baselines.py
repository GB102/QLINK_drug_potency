"""Practical screening baselines: random ranking and a tuned tree ensemble on full fingerprints.

The tree ensemble gets the full ECFP4-2048 + PubChem-881 fingerprint, not the 16-dim compact embedding, so
the comparison shows whether feature compression (not the quantum block) is the binding constraint.
Hyperparameters are chosen on the validation split only.

Run:  python -m qlink_screen.baselines --task potent
"""
import argparse
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import DATA_V2, RESULTS
from .metrics import binary_report, multiclass_report

RUNS = RESULTS / "runs"
GRID = [{"n_estimators": n, "max_features": mf, "min_samples_leaf": ml}
        for n in (300, 600) for mf in ("sqrt", 0.05) for ml in (1, 3)]


def features(kind="fp"):
    d = np.load(DATA_V2 / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    X = np.hstack([d["ecfp4"], d["pubchem"]]).astype(np.float32) if kind == "fp" else d["compact"]
    return X, split, d


def run_random(task, seeds=(0, 1, 2)):
    X, split, d = features()
    y = (d["potent"] if task == "potent" else d["y3"]).astype(int)
    yte = y[split == "test"]
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        if task == "potent":
            sc = rng.random(len(yte))
            out.append(binary_report(sc, yte))
        else:
            p = rng.random((len(yte), 3))
            out.append(multiclass_report(p / p.sum(1, keepdims=True), yte))
    RUNS.mkdir(parents=True, exist_ok=True)
    for s, r in zip(seeds, out):
        (RUNS / f"{task}__random__d0__s{s}.json").write_text(json.dumps(
            {"task": task, "model": "random", "n_blocks": 0, "seed": s, "features": "none",
             "selection_metric": "none", "test_precomputed": r, "params": {"total": 0, "quantum": 0},
             "wall_clock_s": 0.0}, indent=2))
    key = "EF" if task == "potent" else "balanced_accuracy"
    print(f"random {key}: {np.mean([r[key] for r in out]):.3f}")


def run_rf(task, seeds=(0, 1, 2)):
    X, split, d = features()
    y = (d["potent"] if task == "potent" else d["y3"]).astype(int)
    tr, va = split == "train", split == "val"
    sel = "PR_AUC" if task == "potent" else "macro_F1"
    rep = binary_report if task == "potent" else multiclass_report
    prob = (lambda m, Z: m.predict_proba(Z)[:, 1]) if task == "potent" else (lambda m, Z: m.predict_proba(Z))

    best, best_cfg = -np.inf, None
    for cfg in GRID:
        m = RandomForestClassifier(**cfg, class_weight="balanced", n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        v = rep(prob(m, X[va]), y[va])[sel]
        if v > best:
            best, best_cfg = v, cfg
    print(f"rf_fp tuned on validation {sel}={best:.4f} with {best_cfg}")

    RUNS.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        t0 = time.perf_counter()
        m = RandomForestClassifier(**best_cfg, class_weight="balanced", n_jobs=-1, random_state=s).fit(X[tr], y[tr])
        wall = time.perf_counter() - t0
        run = {"task": task, "model": "rf_fp", "n_blocks": 0, "seed": s,
               "features": "ecfp4_2048 + pubchem_881 (full fingerprints)", "selection_metric": sel,
               "best_val": float(rep(prob(m, X[va]), y[va])[sel]), "config": best_cfg,
               "val": rep(prob(m, X[va]), y[va]),
               "params": {"total": int(sum(t.tree_.node_count for t in m.estimators_)), "quantum": 0},
               "wall_clock_s": round(wall, 2), "grid_size": len(GRID)}
        (RUNS / f"{task}__rf_fp__d0__s{s}.json").write_text(json.dumps(run, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="potent", choices=["potent", "class3"])
    a = ap.parse_args()
    run_random(a.task)
    run_rf(a.task)


if __name__ == "__main__":
    main()
