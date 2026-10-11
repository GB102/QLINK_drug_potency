"""Aggregate validation runs, select depth, then evaluate the locked test split once.

    python -m qlink_screen.evaluate              # validation summary + selection (test untouched)
    python -m qlink_screen.evaluate --final      # evaluates the test split for the selected models
    python -m qlink_screen.evaluate --shots      # finite-shot inference with Aer on the selected models

`--final` writes results/final_test.json. Re-running it overwrites that file but never changes the
selection, which is fixed by validation only.
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from . import RESULTS
from .metrics import binary_report, ef_curve, multiclass_report, paired_delta, rank
from .models import build_model
from .train import TASKS, load

RUNS = RESULTS / "runs"
QUANTUM = ["be4", "van4", "van5", "qlink", "qlink_sparse", "qlink_adaptive"]
BASELINES = ["random", "rf_fp", "dnn_fp", "mlp", "mlp_bneck"]


def read_runs(task):
    out = defaultdict(list)
    for f in sorted(RUNS.glob(f"{task}__*.json")):
        r = json.loads(f.read_text())
        out[(r["model"], r["n_blocks"])].append(r)
    return out


def summarize(task):
    runs = read_runs(task)
    sel = TASKS[task]["select"]
    rows = []
    for (model, depth), rs in sorted(runs.items()):
        if model == "random":
            continue
        v = [r["best_val"] for r in rs]
        rows.append({"model": model, "depth": depth, "n_seeds": len(rs),
                     f"val_{sel}_mean": round(float(np.mean(v)), 4),
                     f"val_{sel}_sd": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4),
                     "epochs_mean": round(float(np.mean([r.get("epochs_run", 0) for r in rs])), 1),
                     "wall_s_mean": round(float(np.mean([r.get("wall_clock_s", 0) for r in rs])), 1),
                     "params": rs[0]["params"]["total"],
                     "quantum_params": rs[0]["params"]["quantum"],
                     "ionq_2q_gates": rs[0].get("resources", {}).get("ionq_native", {}).get("two_qubit_gates")})
    chosen = {}
    for m in {r["model"] for r in rows}:
        cand = [r for r in rows if r["model"] == m]
        chosen[m] = max(cand, key=lambda r: r[f"val_{sel}_mean"])["depth"]
    return rows, chosen


def test_scores(task, model, depth, seed):
    """Score the locked test split from a saved checkpoint."""
    ck = torch.load(RUNS / f"{task}__{model}__d{depth}__s{seed}.pt", map_location="cpu", weights_only=False)
    feats = ck["run"]["features"]
    data = load(task, "pubchem" if feats.startswith("pubchem") else "compact")
    Xte, yte = data["test"]
    m = build_model(model, depth, Xte.shape[1], TASKS[task]["n_out"], seed=seed)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    with torch.no_grad():
        out = torch.cat([m(Xte[i:i + 256]) for i in range(0, len(Xte), 256)])
    probs = torch.sigmoid(out.squeeze(-1)).numpy() if task == "potent" else torch.softmax(out, 1).numpy()
    return probs, yte.numpy()


def rf_test_scores(task, seed):
    from sklearn.ensemble import RandomForestClassifier
    run = json.loads((RUNS / f"{task}__rf_fp__d0__s{seed}.json").read_text())
    d = np.load(RESULTS.parent / "data" / "v2" / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    X = np.hstack([d["ecfp4"], d["pubchem"]]).astype(np.float32)
    y = (d["potent"] if task == "potent" else d["y3"]).astype(int)
    m = RandomForestClassifier(**run["config"], class_weight="balanced", n_jobs=-1,
                               random_state=seed).fit(X[split == "train"], y[split == "train"])
    p = m.predict_proba(X[split == "test"])
    return (p[:, 1] if task == "potent" else p), y[split == "test"]


def random_test_scores(task, seed):
    d = np.load(RESULTS.parent / "data" / "v2" / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    y = (d["potent"] if task == "potent" else d["y3"]).astype(int)[split == "test"]
    rng = np.random.default_rng(seed)
    if task == "potent":
        return rng.random(len(y)), y
    p = rng.random((len(y), 3))
    return p / p.sum(1, keepdims=True), y


def final(task, seeds=(0, 1, 2)):
    rows, chosen = summarize(task)
    rep = binary_report if task == "potent" else multiclass_report
    key = "EF" if task == "potent" else "balanced_accuracy"
    per_model, curves = {}, {}
    for model, depth in sorted(chosen.items()):
        got = []
        for s in seeds:
            try:
                if model == "rf_fp":
                    p, y = rf_test_scores(task, s)
                elif model == "random":
                    p, y = random_test_scores(task, s)
                else:
                    p, y = test_scores(task, model, depth, s)
            except FileNotFoundError:
                continue
            got.append({"seed": s, **rep(p, y)})
            if task == "potent":
                curves.setdefault(model, []).append(ef_curve(p, y))
        if got:
            per_model[model] = {"depth": depth, "seeds": got,
                                "mean": {k: round(float(np.mean([g[k] for g in got])), 4)
                                         for k in got[0] if isinstance(got[0][k], (int, float))},
                                "sd": {k: round(float(np.std([g[k] for g in got], ddof=1)), 4)
                                       for k in got[0] if isinstance(got[0][k], (int, float))}
                                if len(got) > 1 else {}}
    # add random even if no checkpoint-based entry exists
    if "random" not in per_model:
        got = [{"seed": s, **rep(*random_test_scores(task, s))} for s in seeds]
        per_model["random"] = {"depth": 0, "seeds": got,
                               "mean": {k: round(float(np.mean([g[k] for g in got])), 4)
                                        for k in got[0] if isinstance(got[0][k], (int, float))}}

    deltas = {}
    for a, b in [("qlink", "van4"), ("qlink", "van5"), ("qlink", "be4"), ("qlink_sparse", "qlink"),
                 ("qlink", "mlp_bneck"), ("qlink", "rf_fp"), ("rf_fp", "qlink")]:
        if a in per_model and b in per_model:
            sa = {g["seed"]: g for g in per_model[a]["seeds"]}
            sb = {g["seed"]: g for g in per_model[b]["seeds"]}
            common = sorted(set(sa) & set(sb))
            if common:
                deltas[f"{a}_minus_{b}"] = {m: paired_delta([sa[s][m] for s in common],
                                                            [sb[s][m] for s in common])
                                            for m in ([key, "PR_AUC", "ROC_AUC"] if task == "potent"
                                                      else [key, "macro_F1"])}
    # Average each EF curve over seeds, so the plotted curve matches the table's mean, not one seed.
    for model, per_seed in list(curves.items()):
        base = per_seed[0]
        curves[model] = [{"K": p["K"], "frac": p["frac"], "n_pool": p["n_pool"],
                          "n_actives_pool": p["n_actives_pool"], "n_seeds": len(per_seed),
                          **{k: float(np.mean([c[i][k] for c in per_seed]))
                             for k in ("EF", "precision@K", "recall@K", "hits@K")}}
                         for i, p in enumerate(base)]

    # Matched-depth comparisons: the selected-configuration deltas above let depth differ, which
    # confounds the messenger question with the depth question. These hold depth fixed.
    matched = {}
    for depth in (4, 8):
        scores = {}
        for m in ("van4", "van5", "qlink", "qlink_sparse", "be4"):
            try:
                scores[m] = [rep(*test_scores(task, m, depth, s)) for s in seeds]
            except FileNotFoundError:
                continue
        for a, b in [("qlink", "van4"), ("qlink_sparse", "qlink"), ("qlink_sparse", "van4")]:
            if a in scores and b in scores:
                matched[f"d{depth}__{a}_minus_{b}"] = {
                    m: paired_delta([g[m] for g in scores[a]], [g[m] for g in scores[b]])
                    for m in ([key, "PR_AUC", "ROC_AUC"] if task == "potent" else [key, "macro_F1"])}
        matched[f"d{depth}__means"] = {m: {k: round(float(np.mean([g[k] for g in v])), 4)
                                           for k in (key, "PR_AUC", "ROC_AUC") if k in v[0]}
                                       for m, v in scores.items()}

    out = {"task": task, "selection": "depth chosen by mean validation " + TASKS[task]["select"],
           "chosen_depths": chosen, "validation_summary": rows,
           "test": per_model, "paired_deltas": deltas, "matched_depth": matched,
           "ef_curves": curves, "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    (RESULTS / f"final_test_{task}.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== {task}: locked test split, {len(per_model)} models ===")
    if task == "potent":
        hdr = f"{'model':<16}{'d':>2}{'EF@10%':>9}{'prec@K':>9}{'hits@K':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'2q':>6}"
    else:
        hdr = f"{'model':<16}{'d':>2}{'bal.acc':>9}{'macroF1':>9}{'acc':>8}{'ROC-AUC':>9}"
    print(hdr)
    order = sorted(per_model, key=lambda m: -per_model[m]["mean"][key])
    for m in order:
        e = per_model[m]
        g2 = next((r["ionq_2q_gates"] for r in rows if r["model"] == m and r["depth"] == e["depth"]), None)
        if task == "potent":
            print(f"{m:<16}{e['depth']:>2}{e['mean']['EF']:>9.2f}{e['mean']['precision@K']:>9.3f}"
                  f"{e['mean']['hits@K']:>8.1f}{e['mean']['PR_AUC']:>9.3f}{e['mean']['ROC_AUC']:>9.3f}"
                  f"{(g2 if g2 is not None else 0):>6}")
        else:
            print(f"{m:<16}{e['depth']:>2}{e['mean']['balanced_accuracy']:>9.3f}{e['mean']['macro_F1']:>9.3f}"
                  f"{e['mean']['accuracy']:>8.3f}{e['mean']['ROC_AUC_ovr']:>9.3f}")
    print("\npaired per-seed differences:")
    for k, v in deltas.items():
        d = v[key]
        extra = f" boot95 [{d['boot95'][0]:+.3f}, {d['boot95'][1]:+.3f}]" if "boot95" in d else ""
        print(f"  {k:<26}{key} {d['mean']:+.3f}{extra}  wins {d.get('wins', '-')}/{d['n_seeds']}")
    return out


def shots_eval(task="potent", models=("qlink", "qlink_sparse", "van4"), shot_budgets=(512, 2048), seed=0):
    """Finite-shot inference on the selected checkpoints, using Aer's ideal sampler.

    This is ideal sampling, not a device noise model: it isolates the cost of finite statistics.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from .circuits import build as build_circuit
    _, chosen = summarize(task)
    d = np.load(RESULTS.parent / "data" / "v2" / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    Xte = d["compact"][split == "test"].astype(np.float32)
    yte = d["potent"][split == "test"].astype(int)
    sim = AerSimulator()
    out = {}
    for model in models:
        depth = chosen.get(model)
        if depth is None:
            continue
        ck = torch.load(RUNS / f"{task}__{model}__d{depth}__s{seed}.pt", map_location="cpu",
                        weights_only=False)
        sd = ck["state_dict"]
        qc, xs, ws, obs = build_circuit(model, depth)
        n_data = len(obs)
        angles = np.tanh(Xte @ sd["pre.weight"].numpy().T + sd["pre.bias"].numpy()) * (np.pi / 2)
        wv = sd["weights"].numpy()
        pw, pb = sd["post.weight"].numpy(), sd["post.bias"].numpy()
        exact, _ = test_scores(task, model, depth, seed)
        entry = {"depth": depth, "exact": binary_report(exact, yte)}
        for shots in shot_budgets:
            t0 = time.perf_counter()
            circs = []
            for a in angles:
                b = qc.assign_parameters({**{p: float(v) for p, v in zip(xs, a)},
                                          **{p: float(v) for p, v in zip(ws, wv)}})
                m = b.copy()
                m.measure_all()
                circs.append(transpile(m, sim))
            res = sim.run(circs, shots=shots, seed_simulator=1234).result()
            z = np.zeros((len(circs), n_data))
            for i in range(len(circs)):
                for bits, c in res.get_counts(i).items():
                    bits = bits.replace(" ", "")
                    for q in range(n_data):
                        z[i, q] += c * (1 - 2 * int(bits[len(bits) - 1 - q]))
            z /= shots
            sc = 1 / (1 + np.exp(-(z @ pw.T + pb).ravel()))
            r = binary_report(sc, yte)
            r["wall_clock_s"] = round(time.perf_counter() - t0, 1)
            r["spearman_vs_exact"] = float(np.corrcoef(rank(sc).argsort(), rank(exact).argsort())[0, 1])
            r["topK_overlap_vs_exact"] = float(len(set(rank(sc)[:r["K"]]) & set(rank(exact)[:r["K"]])) / r["K"])
            entry[f"shots_{shots}"] = r
            print(f"{model} d{depth} {shots:>5} shots: EF@10%={r['EF']:.2f} (exact {entry['exact']['EF']:.2f}) "
                  f"top-K overlap {r['topK_overlap_vs_exact']:.2f}  {r['wall_clock_s']}s")
        out[model] = entry
    (RESULTS / "shots_eval.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="potent", choices=list(TASKS))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--shots", action="store_true")
    a = ap.parse_args()
    if a.shots:
        shots_eval(a.task)
    elif a.final:
        final(a.task)
    else:
        rows, chosen = summarize(a.task)
        sel = TASKS[a.task]["select"]
        print(f"{'model':<16}{'d':>3}{'seeds':>6}{'val ' + sel:>14}{'sd':>8}{'epochs':>8}{'s':>7}{'2q':>6}")
        for r in sorted(rows, key=lambda r: -r[f"val_{sel}_mean"]):
            print(f"{r['model']:<16}{r['depth']:>3}{r['n_seeds']:>6}{r[f'val_{sel}_mean']:>14.4f}"
                  f"{r[f'val_{sel}_sd']:>8.4f}{r['epochs_mean']:>8.1f}{r['wall_s_mean']:>7.1f}"
                  f"{(r['ionq_2q_gates'] or 0):>6}")
        print("\nselected depth per model (validation only):", chosen)
