"""The three evidence figures. Every plotted value is read from a saved result file and also written
next to the figure as a CSV table view, so identity never rests on colour alone and each number traces
back to a run. The CSVs carry every model; the plots carry the subset that answers the question.

    python -m qlink_screen.figures

Inputs : results/final_test_potent.json, results/runs/*.json, results/shots_eval.json
Outputs: figures/fig1_enrichment_vs_k.png, fig2_gradients_vs_depth.png, fig3_quality_vs_cost.png (+ .csv)
"""
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from . import FIGURES, RESULTS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"

# Quantum variants take categorical slots in fixed order; classical references are neutral + dashed,
# a composite encoding that keeps the two families apart by hue AND line style. Colour follows the
# model, identically in every figure, and never changes when a figure shows fewer series.
STYLE = {
    "qlink":        dict(color="#2a78d6", marker="o", ls="-",  label="Q-LINK (4+1)"),
    "qlink_sparse": dict(color="#eb6834", marker="s", ls="-",  label="Q-LINK sparse (4+1)"),
    "van4":         dict(color="#1baf7a", marker="^", ls="-",  label="ordinary VQC (4q)"),
    "van5":         dict(color="#eda100", marker="D", ls="-",  label="ordinary VQC (5q control)"),
    "be4":          dict(color="#e87ba4", marker="v", ls="-",  label="team's BasicEntangler (4q)"),
    "rf_fp":        dict(color="#52514e", marker="D", ls="--", label="random forest, full FP"),
    "dnn_fp":       dict(color="#6f6e69", marker="*", ls="--", label="team's DNN, full FP"),
    "mlp":          dict(color="#8a8a84", marker="P", ls="--", label="classical MLP, same features"),
    "mlp_bneck":    dict(color="#a5a49e", marker="X", ls="--", label="classical 4-wide bottleneck"),
    "random":       dict(color="#9c9b95", marker="",  ls=":",  label="random picking"),
}


def style(ax, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title + ("\n" + subtitle if subtitle else ""), color=INK, fontsize=11.5,
                     loc="left", pad=10)


def _save(fig, name, table):
    FIGURES.mkdir(exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    (FIGURES / f"{name}.csv").write_text("\n".join(table) + "\n")
    print(f"wrote figures/{name}.png and .csv")


def fig1_enrichment():
    """Screening enrichment versus assay budget K -- the primary endpoint."""
    d = json.loads((RESULTS / "final_test_potent.json").read_text())
    curves, test = d["ef_curves"], d["test"]
    shown = [m for m in ("qlink", "qlink_sparse", "van4", "rf_fp", "mlp") if m in curves]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    style(ax, "assay budget K (compounds selected from the held-out pool)", "enrichment factor EF@K",
          "Screening enrichment against the assay budget",
          "held-out scaffold-split test pool, HIV-1 protease; potent = IC50 ≤ 10 nM; mean of 3 seeds")
    rows = ["# every model, including those not drawn; EF curve averaged over 3 seeds",
            "model,K,frac,hits@K,precision@K,EF@K,recall@K"]
    for m, c in curves.items():
        for p in c:
            rows.append(f"{m},{p['K']},{p['frac']:.2f},{p['hits@K']:.2f},{p['precision@K']:.4f},"
                        f"{p['EF']:.4f},{p['recall@K']:.4f}")

    k10 = min(curves[shown[0]], key=lambda p: abs(p["frac"] - 0.10))["K"]
    for m in shown:
        c = curves[m]
        s = STYLE[m]
        ax.plot([p["K"] for p in c], [p["EF"] for p in c], lw=2, color=s["color"], ls=s["ls"],
                marker=s["marker"], markersize=5, markevery=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, solid_joinstyle="round", solid_capstyle="round",
                label=s["label"], zorder=3)
    ax.axhline(1.0, color=INK_2, lw=1, ls="--", zorder=2)
    ax.annotate("random picking (EF = 1)", xy=(185, 1.0), xytext=(0, 6), textcoords="offset points",
                color=INK_2, fontsize=8.5)
    ax.axvline(k10, color=INK_2, lw=1, ls=":", zorder=2)

    # Direct labels at the pre-declared budget, for the two series the comparison is about.
    for m, dy in (("mlp", 16), ("qlink", -36)):
        if m not in curves:
            continue
        p = min(curves[m], key=lambda p: abs(p["frac"] - 0.10))
        ax.plot([p["K"]], [p["EF"]], marker=STYLE[m]["marker"], markersize=10, color=STYLE[m]["color"],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5, ls="none")
        ax.annotate(f"{STYLE[m]['label']}\nEF {p['EF']:.2f} at K={p['K']}", xy=(p["K"], p["EF"]),
                    xytext=(14, dy), textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold")
    ax.annotate(f"pre-declared budget K = {k10} (top 10%)", xy=(k10, 2.72), xytext=(6, 0),
                textcoords="offset points", color=INK_2, fontsize=8.5)
    ax.set_xlim(0, 365)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right", borderaxespad=0.4)
    fig.text(0.0, -0.045, "All ten models are in fig1_enrichment_vs_k.csv; the five drawn here are the "
                          "two Q-LINK variants, the matched ordinary circuit,\nand the two strongest "
                          "classical baselines.", color=INK_2, fontsize=8.5, ha="left")
    _save(fig, "fig1_enrichment_vs_k", rows)


def fig2_gradients():
    """Trainability versus circuit depth: gradient magnitude, variance, and circuit-output sensitivity."""
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted((RESULTS / "runs").glob("potent__*.json")):
        r = json.loads(f.read_text())
        g = r.get("gradients", {})
        if "quantum_grad_abs_mean" in g:
            runs[r["model"]][r["n_blocks"]].append(g)
    shown = [m for m in ("be4", "van4", "qlink", "qlink_sparse") if m in runs]
    panels = [("quantum_grad_abs_mean", "mean |∂loss/∂θ| over circuit parameters", "Gradient magnitude"),
              ("quantum_grad_var", "variance of circuit-parameter gradients", "Gradient variance"),
              ("quantum_out_jacobian_abs_mean", "mean |∂⟨Z⟩/∂θ|", "Circuit-output sensitivity")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    rows = ["# van5 is omitted from the plot: its idle qubit makes it numerically identical to van4",
            "model,n_blocks,metric,mean,sd,n_initializations"]
    for m in runs:
        for dep in sorted(runs[m]):
            for key, _, _ in panels:
                v = [g[key] for g in runs[m][dep]]
                rows.append(f"{m},{dep},{key},{np.mean(v):.6g},"
                            f"{np.std(v, ddof=1) if len(v) > 1 else 0:.6g},"
                            f"{sum(g['n_init'] for g in runs[m][dep])}")
    for ax, (key, ylab, title) in zip(axes, panels):
        style(ax, "operation blocks (circuit depth)", ylab, title)
        for j, m in enumerate(shown):
            deps = sorted(runs[m])
            mean = [float(np.mean([g[key] for g in runs[m][d]])) for d in deps]
            sd = [float(np.std([g[key] for g in runs[m][d]], ddof=1)) for d in deps]
            s = STYLE[m]
            ax.errorbar(deps, mean, yerr=sd, lw=2, color=s["color"], marker=s["marker"], markersize=9,
                        markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, elinewidth=1,
                        ecolor=s["color"], label=s["label"], zorder=3)
            if len(deps) == 2:
                # nudge alternate series apart so the ratio labels never collide
                ax.annotate(f"×{mean[1] / mean[0]:.2f}", xy=(deps[1], mean[1]),
                            xytext=(10, 6 if j % 2 else -12), textcoords="offset points",
                            color=s["color"], fontsize=9, fontweight="bold")
        ax.set_xticks(sorted({d for m in shown for d in runs[m]}))
        ax.set_xlim(3.4, 9.2)
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Trainability against circuit depth  —  fixed diagnostic batch of 64, 24 independent "
                 "initializations per point, before any training\n"
                 "×N is the depth-8 value relative to depth 4: the ordinary circuits attenuate with "
                 "depth, the messenger circuits do not.",
                 color=INK, fontsize=11.5, x=0.004, ha="left", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_gradients_vs_depth", rows)


def fig3_quality_vs_cost():
    """Predictive quality against compiled two-qubit gate cost, and against the shot budget."""
    d = json.loads((RESULTS / "final_test_potent.json").read_text())
    test, vsum, matched = d["test"], d["validation_summary"], d.get("matched_depth", {})
    gates = {(r["model"], r["depth"]): r["ionq_2q_gates"] for r in vsum}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    rows = ["panel,model,n_blocks,x_value,EF@10%"]

    ax = axes[0]
    style(ax, "two-qubit gates after compilation for IonQ's native set", "EF@10% on the held-out test pool",
          "Quality against compiled gate cost",
          "each quantum model at both depths, mean over 3 seeds")
    series = defaultdict(list)
    for depth in (4, 8):
        means = matched.get(f"d{depth}__means", {})
        for m, v in means.items():
            g = gates.get((m, depth))
            if g is not None and m in STYLE:
                series[m].append((g, v["EF"], depth))
    for m, pts in series.items():
        if m == "van5":
            continue                      # numerically identical to van4; kept in the CSV
        pts.sort()
        s = STYLE[m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2, ls=s["ls"], color=s["color"],
                marker=s["marker"], markersize=10, markeredgecolor=SURFACE, markeredgewidth=2,
                label=s["label"], zorder=3)
        for g, ef, dep in pts:
            ax.annotate(f"{dep}", xy=(g, ef), xytext=(0, 13), textcoords="offset points",
                        color=s["color"], fontsize=8.5, ha="center", fontweight="bold")
    for m, pts in series.items():
        for g, ef, dep in pts:
            rows.append(f"gates,{m},{dep},{g},{ef:.4f}")
    for m in ("rf_fp", "mlp"):
        if m in test:
            ef = test[m]["mean"]["EF"]
            ax.axhline(ef, color=STYLE[m]["color"], lw=1.6, ls=STYLE[m]["ls"], zorder=2)
            ax.annotate(f"{STYLE[m]['label']}: EF {ef:.2f}  (no quantum gates)", xy=(90, ef),
                        xytext=(0, 4), textcoords="offset points", color=STYLE[m]["color"],
                        fontsize=8.5, ha="right")
            rows.append(f"gates,{m},0,0,{ef:.4f}")
    ax.annotate("depth labelled above each point", xy=(0.02, 0.03), xycoords="axes fraction",
                color=INK_2, fontsize=8.5)
    ax.set_xlim(5, 95)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center right", borderaxespad=0.4)

    ax = axes[1]
    style(ax, "shots per compound (ideal sampling, Aer)", "EF@10% on the held-out test pool",
          "Quality against the shot budget",
          "the selected checkpoints, frozen; dashed line = that model's exact statevector value")
    try:
        sh = json.loads((RESULTS / "shots_eval.json").read_text())
    except FileNotFoundError:
        sh = {}
    budgets = sorted({int(k.split("_")[1]) for e in sh.values() for k in e if k.startswith("shots_")})
    for m, e in sh.items():
        s = STYLE[m]
        y = [e[f"shots_{b}"]["EF"] for b in budgets]
        ax.plot(budgets, y, lw=2, color=s["color"], marker=s["marker"], markersize=10,
                markeredgecolor=SURFACE, markeredgewidth=2, label=s["label"], zorder=3)
        ax.axhline(e["exact"]["EF"], color=s["color"], lw=1, ls="--", zorder=2, alpha=0.8)
        for b in budgets:
            rows.append(f"shots,{m},{e['depth']},{b},{e[f'shots_{b}']['EF']:.4f}")
        rows.append(f"shots,{m},{e['depth']},exact,{e['exact']['EF']:.4f}")
    if sh:
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlim(budgets[0] * 0.8, budgets[-1] * 1.25)
        lo = min(min(e["exact"]["EF"], *[e[f"shots_{b}"]["EF"] for b in budgets]) for e in sh.values())
        hi = max(max(e["exact"]["EF"], *[e[f"shots_{b}"]["EF"] for b in budgets]) for e in sh.values())
        ax.set_ylim(lo - 0.04, hi + (hi - lo) * 0.42)      # headroom so the legend clears every mark
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left", borderaxespad=0.6,
                  ncol=3, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    _save(fig, "fig3_quality_vs_cost", rows)


if __name__ == "__main__":
    fig1_enrichment()
    fig2_gradients()
    fig3_quality_vs_cost()
