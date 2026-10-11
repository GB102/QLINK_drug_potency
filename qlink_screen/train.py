"""Training, gradient diagnostics and resource accounting for one (model, depth, seed, task) run.

Run IDs: results/runs/<task>__<model>__d<blocks>__s<seed>.json  (+ .pt checkpoint)
Nothing here writes into the team's original directories.

Usage:
    python -m qlink_screen.train --models van4 qlink --depths 4 8 --seeds 0 1 2
    python -m qlink_screen.train --task class3 --models rf_fp dnn_fp
"""
import argparse
import json
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from . import DATA_V2, RESULTS
from .metrics import binary_report, multiclass_report
from .models import build_model, count_parameters
from .resources import circuit_resources

RUNS = RESULTS / "runs"
TASKS = {"potent": {"n_out": 1, "select": "PR_AUC"}, "class3": {"n_out": 3, "select": "macro_F1"}}


def load(task, features="compact"):
    d = np.load(DATA_V2 / "features.npz", allow_pickle=True)
    split = d["split"].astype(str)
    X = d[features].astype(np.float32)
    if features == "pubchem":
        keep = X[split == "train"].var(0) > 0.01
        X = X[:, keep]
    y = d["potent" if task == "potent" else "y3"].astype(np.int64)
    return {s: (torch.tensor(X[split == s]), torch.tensor(y[split == s])) for s in ("train", "val", "test")}


def _loss_fn(task, y_train):
    if task == "potent":
        pos = float((y_train == 1).sum())
        w = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)])
        return nn.BCEWithLogitsLoss(pos_weight=w), lambda o: torch.sigmoid(o.squeeze(-1))
    counts = np.bincount(y_train.numpy(), minlength=3)
    w = torch.tensor(len(y_train) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=w), lambda o: torch.softmax(o, 1)


def _score(task, model, X, to_prob, bs=256):
    model.eval()
    with torch.no_grad():
        out = torch.cat([to_prob(model(X[i:i + bs])) for i in range(0, len(X), bs)])
    return out.numpy()


def report(task, probs, y):
    return binary_report(probs, y) if task == "potent" else multiclass_report(probs, y)


def gradient_diagnostics(task, model, X, y, loss_fn, n_init=8, batch=64, seed=0):
    """Gradient magnitude/variance on a fixed diagnostic batch across independent initializations.

    Quantum-circuit gradients and classical-head gradients are kept separate, and the quantum-output
    Jacobian is reported so circuit insensitivity can be told apart from head/loss saturation.
    """
    g = torch.Generator().manual_seed(1234)
    idx = torch.randperm(len(X), generator=g)[:batch]
    xb, yb = X[idx], y[idx]
    yt = yb.float().unsqueeze(-1) if task == "potent" else yb
    qn, cn, jac = [], [], []
    for k in range(n_init):
        m = build_model(model.model_id if hasattr(model, "model_id") else "mlp",
                        getattr(model, "n_blocks", 4), X.shape[1],
                        TASKS[task]["n_out"], seed=seed * 1000 + k)
        m.train()
        loss_fn(m(xb), yt).backward()
        qp = [p.grad.reshape(-1) for p in m.quantum_parameters() if p.grad is not None]
        cp = [p.grad.reshape(-1) for p in m.classical_parameters() if p.grad is not None]
        if qp:
            qn.append(torch.cat(qp).numpy())
        cn.append(torch.cat(cp).abs().mean().item())
        if hasattr(m, "quantum_out"):
            m.zero_grad()
            qo = m.quantum_out(xb)
            jac.append(torch.autograd.grad(qo.sum(), m.quantum_parameters(), retain_graph=False)[0]
                       .abs().mean().item())
    out = {"n_init": n_init, "batch": batch,
           "classical_grad_abs_mean": float(np.mean(cn))}
    if qn:
        a = np.stack(qn)
        out.update({"quantum_grad_abs_mean": float(np.abs(a).mean()),
                    "quantum_grad_var": float(a.var()),
                    "quantum_grad_var_across_init": float(a.mean(1).var()),
                    "quantum_grad_max_abs": float(np.abs(a).max()),
                    "quantum_out_jacobian_abs_mean": float(np.mean(jac))})
    return out


def train_one(task, model_id, n_blocks=4, seed=0, features="compact", epochs=80, patience=15,
              lr=1e-2, batch=64, diagnostics=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load(task, features)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    n_out = TASKS[task]["n_out"]
    model = build_model(model_id, n_blocks, Xtr.shape[1], n_out, seed=seed)
    loss_fn, to_prob = _loss_fn(task, ytr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    sel = TASKS[task]["select"]

    diag = gradient_diagnostics(task, model, Xtr, ytr, loss_fn, seed=seed) if diagnostics else {}
    g = torch.Generator().manual_seed(seed)
    hist, best, best_state, best_epoch, wait, steps = [], -np.inf, None, 0, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(perm) - batch + 1, batch):
            b = perm[i:i + batch]
            yb = ytr[b].float().unsqueeze(-1) if task == "potent" else ytr[b]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            steps += 1
        va = report(task, _score(task, model, Xva, to_prob), yva.numpy())
        tr_loss = tot / max(1, (len(perm) // batch) * batch)
        with torch.no_grad():
            yv = yva.float().unsqueeze(-1) if task == "potent" else yva
            model.eval()
            va_loss = float(loss_fn(model(Xva), yv))
        hist.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                     **{k: va[k] for k in (sel, "EF" if task == "potent" else "balanced_accuracy")}})
        sched.step(va[sel])
        if va[sel] > best:
            best, best_epoch, wait = va[sel], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    wall = time.perf_counter() - t0
    model.load_state_dict(best_state)

    run = {"task": task, "model": model_id, "n_blocks": n_blocks, "seed": seed, "features": features,
           "selection_metric": sel, "best_val": float(best), "best_epoch": best_epoch,
           "epochs_run": len(hist), "optimizer_steps": steps,
           "params": count_parameters(model), "wall_clock_s": round(wall, 2),
           "s_per_epoch": round(wall / max(1, len(hist)), 3),
           "val": report(task, _score(task, model, Xva, to_prob), yva.numpy()),
           "history": hist, "gradients": diag,
           "hardware": {"cpu": platform.processor() or platform.machine(), "torch": torch.__version__},
           "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte))}
    if hasattr(model, "circuit"):
        run["resources"] = circuit_resources(model.circuit)
        run["circuit_evaluations"] = steps * batch  # one batched statevector call per minibatch
    RUNS.mkdir(parents=True, exist_ok=True)
    rid = f"{task}__{model_id}__d{n_blocks}__s{seed}"
    (RUNS / f"{rid}.json").write_text(json.dumps(run, indent=2))
    torch.save({"state_dict": best_state, "run": {k: v for k, v in run.items() if k != "history"}},
               RUNS / f"{rid}.pt")
    print(f"{rid}: val {sel}={best:.4f} @epoch {best_epoch} | {wall:.1f}s | params {run['params']}")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="potent", choices=list(TASKS))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--depths", nargs="+", type=int, default=[4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--features", default="compact")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--no-diagnostics", action="store_true")
    a = ap.parse_args()
    for m in a.models:
        depths = a.depths if m in ("be4", "van4", "van5", "qlink", "qlink_sparse", "qlink_adaptive") else [0]
        for d in depths:
            for s in a.seeds:
                feats = "pubchem" if m == "dnn_fp" else a.features
                train_one(a.task, m, d, s, feats, epochs=a.epochs, diagnostics=not a.no_diagnostics)


if __name__ == "__main__":
    main()
