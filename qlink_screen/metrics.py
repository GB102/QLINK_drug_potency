"""Screening metrics. EF@K is the pre-declared primary endpoint.

Ties: scores are sorted with a stable descending sort on (-score, index), so every model breaks ties by the
same fixed test-pool order. `ties_at_cut` records how many compounds share the score at the cut, so a
tie-sensitive result is visible rather than hidden.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)


def rank(scores):
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))


def enrichment(scores, labels, frac=0.10):
    """EF@frac = (hit rate in the top K) / (hit rate in the whole pool)."""
    labels = np.asarray(labels).astype(int)
    n, n_act = len(labels), int(labels.sum())
    k = max(1, int(round(frac * n)))
    order = rank(scores)
    top = labels[order[:k]]
    base = n_act / n
    s = np.asarray(scores, dtype=float)[order]
    return {"K": k, "frac": frac, "n_pool": n, "n_actives_pool": n_act,
            "hits@K": int(top.sum()), "precision@K": float(top.mean()),
            "recall@K": float(top.sum() / n_act) if n_act else float("nan"),
            "EF": float(top.mean() / base) if base else float("nan"),
            "EF_max": float(min(1.0, n / k * base) / base) if base else float("nan"),
            "ties_at_cut": int((s == s[k - 1]).sum())}


def ef_curve(scores, labels, fracs=None):
    fracs = fracs if fracs is not None else np.arange(0.02, 1.001, 0.02)
    return [enrichment(scores, labels, f) for f in fracs]


def binary_report(scores, labels, frac=0.10):
    labels = np.asarray(labels).astype(int)
    out = enrichment(scores, labels, frac)
    out["PR_AUC"] = float(average_precision_score(labels, scores))
    out["ROC_AUC"] = float(roc_auc_score(labels, scores))
    out["prevalence"] = float(labels.mean())
    for f in (0.05, 0.20):
        out[f"EF@{int(f * 100)}%"] = enrichment(scores, labels, f)["EF"]
    return out


def multiclass_report(probs, labels):
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(1)
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=probs.shape[1])
    return {"accuracy": float((pred == labels).mean()),
            "majority_class_rate": float(counts.max() / counts.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "macro_F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "ROC_AUC_ovr": float(roc_auc_score(labels, probs / probs.sum(1, keepdims=True),
                                               multi_class="ovr", average="macro"))}


def paired_delta(a, b):
    """Per-seed paired differences a - b with a mean and a bootstrap interval (small-n honest)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return {"mean": float(d.mean()), "n_seeds": len(d)}
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_seeds": int(len(d)),
            "per_seed": [float(v) for v in d],
            "boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((d > 0).sum())}
