"""Figures 7 and 8 -- corrected versions of two three-class sweep charts.

The originals contributed on 11 Oct are kept in `figures/` unchanged. These redraw the same analysis with
the defects recorded in AUDIT section 6 fixed:

  fig7  accuracy per configuration, from values recomputed from each model's saved weights by
        qollab_combined/single_run.py, against the 78.29 % majority-class rate. The original charted the
        classical DNN's 80.36 % for the 4-qubit/2-rep and 4-qubit/4-rep models and transposed 4-rep with
        8-rep, because the notebooks printed a stale variable from the DNN run. The convergence-epoch
        series is dropped rather than redrawn: it was shifted by the same slot, and two y-axes on one
        plot invite reading a relationship the axis alignment invented.

  fig8  the deviation of backend expectation values from exact, which is the informative quantity from
        that run. The original plotted accuracy measured on ONE molecule per model, so every bar was 0 %
        or 100 %, and titled a noise-model simulator run as quantum hardware.

    python -m qlink_screen.figure_sweep
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import FIGURES
from .figures import GRID, INK, INK_2, SURFACE, style

TRIVIAL = 0.7829          # 303 of the 387 test compounds are "active"
CLASSICAL = "#6f6e69"
HYBRID = "#e87ba4"

# Accuracy recomputed from each model's saved weights (qollab_combined/single_run.py, exact statevector).
# The classical DNN value is from its own notebook output.
ACCURACY = [
    ("Classical DNN",       0.8036, CLASSICAL),
    ("QML 4q, 2 rep",       0.7442, HYBRID),
    ("QML 4q, 4 rep",       0.7494, HYBRID),
    ("QML 4q, 8 rep",       0.8036, HYBRID),
    ("QML 8q, 1 rep",       0.7907, HYBRID),
    ("QML 8q, 2 rep",       0.7390, HYBRID),
    ("QML 8q, 4 rep",       0.7778, HYBRID),
    ("QML 16q, 1 rep",      0.7700, HYBRID),
]

# Mean |<Z>_backend - <Z>_exact| per model, as reported by the contributed IonQ run.
# Teammate-reported: these come from that run's console output, not recomputed here.
DEVIATION = [
    ("4 qubits, 2 reps",  0.0618),
    ("8 qubits, 1 rep",   0.0445),
    ("8 qubits, 2 reps",  0.0713),
    ("16 qubits, 1 rep",  0.0546),
]
SHOT_SIGMA_1024 = 1.0 / np.sqrt(1024)


def fig7():
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    style(ax, "", "test accuracy on the three-class task",
          "Three-class accuracy against the only reference that makes it readable",
          "accuracy recomputed from each model's saved weights; 387 held-out compounds")
    x = np.arange(len(ACCURACY))
    for xi, (lab, acc, col) in zip(x, ACCURACY):
        ax.bar(xi, acc, width=0.6, color=col, zorder=3)
        above = acc > TRIVIAL
        ax.annotate(f"{acc * 100:.2f}%", xy=(xi, acc), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK if above else INK_2,
                    fontsize=9.5, fontweight="bold" if above else "normal")
    ax.axhline(TRIVIAL, color=INK, lw=2, ls="--", zorder=4)
    ax.annotate(f"answering \"active\" every time scores {TRIVIAL * 100:.2f}%",
                xy=(-0.45, TRIVIAL), xytext=(0, -14), textcoords="offset points",
                ha="left", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in ACCURACY], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.70, 0.83)
    ax.set_yticks(np.arange(0.70, 0.831, 0.02))
    ax.set_yticklabels([f"{v * 100:.0f}%" for v in np.arange(0.70, 0.831, 0.02)])
    fig.text(0.0, -0.13, "Only two configurations clear the majority-class rate, and both by under "
                         "2.1 points. Accuracy cannot rank these models;\nper-class recall (fig 5) and "
                         "balanced accuracy can.", color=INK_2, fontsize=8.5, ha="left")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / "fig7_multiclass_accuracy_corrected.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    rows = ["# accuracy recomputed from the saved weights by qollab_combined/single_run.py",
            f"# majority-class rate {TRIVIAL}", "model,accuracy,delta_vs_majority"]
    rows += [f"\"{lab}\",{acc:.4f},{acc - TRIVIAL:+.4f}" for lab, acc, _ in ACCURACY]
    (FIGURES / "fig7_multiclass_accuracy_corrected.csv").write_text("\n".join(rows) + "\n")
    print("wrote figures/fig7_multiclass_accuracy_corrected.png and .csv")


def fig8():
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    style(ax, "", "mean |⟨Z⟩ backend − ⟨Z⟩ exact|",
          "How far IonQ's noise model moves the expectation values",
          "IonQ remote simulator (noise model of a QPU, not hardware), 1024 shots")
    x = np.arange(len(DEVIATION))
    for xi, (lab, dz) in zip(x, DEVIATION):
        ax.bar(xi, dz, width=0.55, color=HYBRID, zorder=3)
        ax.annotate(f"{dz:.4f}", xy=(xi, dz), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.axhline(SHOT_SIGMA_1024, color=INK, lw=2, ls="--", zorder=4)
    ax.annotate(f"1-sigma shot noise at 1024 shots = {SHOT_SIGMA_1024:.4f}",
                xy=(-0.42, SHOT_SIGMA_1024), xytext=(0, 7), textcoords="offset points",
                ha="left", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in DEVIATION], fontsize=9.5)
    ax.set_ylim(0, max(d[1] for d in DEVIATION) * 1.35)
    fig.text(0.0, -0.14, "Every model sits above the shot-noise line, so the deviation is the device "
                         "noise model, not finite sampling.\nThe run it comes from measured one molecule "
                         "per model, which is too few for an accuracy; this deviation is what it does "
                         "measure.", color=INK_2, fontsize=8.5, ha="left")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / "fig8_ionq_noise_deviation.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    rows = ["# teammate-reported, from the contributed IonQ remote-simulator run (noise model)",
            f"# 1-sigma shot noise at 1024 shots = {SHOT_SIGMA_1024:.4f}",
            "model,mean_abs_delta_Z,above_shot_noise"]
    rows += [f"\"{lab}\",{dz:.4f},{dz > SHOT_SIGMA_1024}" for lab, dz in DEVIATION]
    (FIGURES / "fig8_ionq_noise_deviation.csv").write_text("\n".join(rows) + "\n")
    print("wrote figures/fig8_ionq_noise_deviation.png and .csv")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fig7()
    fig8()
