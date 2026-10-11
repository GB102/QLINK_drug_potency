"""Figure 4 -- the team's ORIGINAL binary-task results, against the trivial baseline.

These are the pre-correction numbers: the original random stratified split of 2 577 compounds with
"active" = IC50 < 10 uM, which makes the test set 91.7 % positive. They are NOT comparable with the v2
results in figures 1-3 (different compound set, different split, different label). Kept separate on
purpose, and drawn against the only reference that makes accuracy interpretable on a 91.7 %-positive set.

Sources: DNN and 4-rep hybrid from the saved notebook outputs in classical/ and quantum/Binary/;
2-rep hybrid reported by the teammate (balanced accuracy not reported -- see MISSING below).

    python -m qlink_screen.figure_prior_binary
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import FIGURES
from .figures import GRID, INK, INK_2, SURFACE, style

N_TEST, N_ACTIVE = 387, 355
TRIVIAL = N_ACTIVE / N_TEST          # always predict "active"

CLASSICAL = "#6f6e69"
HYBRID = "#e87ba4"                   # the team's BasicEntangler family, as in figures 1-3

MODELS = [
    # label,                  accuracy, balanced accuracy (None = not reported), colour
    ("Classical DNN\n(best epoch 8)",            0.9251, 0.8170, CLASSICAL),
    ("Hybrid QML 4 qubits, 2 reps\n(epoch 25)",  0.9200, None,   HYBRID),
    ("Hybrid QML 4 qubits, 4 reps\n(epoch 23)",  0.8863, 0.8385, HYBRID),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    y = np.arange(len(MODELS))[::-1]

    # ---- panel A: accuracy, against the trivial predictor ----
    ax = axes[0]
    style(ax, "accuracy on the held-out test set", "", "Accuracy says the classical model wins",
          "— but see where the dashed line sits")
    for yi, (lab, acc, _, col) in zip(y, MODELS):
        ax.barh(yi, acc, height=0.46, color=col, zorder=3)
        ax.annotate(f"{acc:.4f}", xy=(acc, yi), xytext=(8, 6), textcoords="offset points",
                    va="center", color=INK, fontsize=10.5, fontweight="bold")
        delta = acc - TRIVIAL
        ax.annotate(f"{delta:+.4f}  ({delta * N_TEST:+.0f} compounds)", xy=(acc, yi),
                    xytext=(8, -8), textcoords="offset points", va="center",
                    color=INK if delta > 0 else "#b3261e", fontsize=9)
    ax.axvline(TRIVIAL, color=INK, lw=2, ls="--", zorder=4)
    ax.set_ylim(-0.75, len(MODELS) - 0.25)
    ax.annotate(f"a model that answers \"active\"\nevery time scores {TRIVIAL:.4f}",
                xy=(TRIVIAL, -0.62), xytext=(-10, 0), textcoords="offset points", ha="right",
                va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0.80, 0.965)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in MODELS], fontsize=9.5)

    # ---- panel B: balanced accuracy, against chance ----
    ax = axes[1]
    style(ax, "balanced accuracy (mean of per-class recalls)", "",
          "Balanced accuracy reverses it",
          "the weakest model on accuracy is the strongest here")
    for yi, (lab, _, bal, col) in zip(y, MODELS):
        if bal is None:
            ax.barh(yi, 0.5, height=0.46, color=SURFACE, edgecolor=col, lw=1.5, hatch="//", zorder=3)
            ax.annotate("not reported — ask for this number", xy=(0.505, yi), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=9,
                        style="italic")
            continue
        ax.barh(yi, bal, height=0.46, color=col, zorder=3)
        ax.annotate(f"{bal:.4f}", xy=(bal, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.axvline(0.5, color=INK, lw=2, ls="--", zorder=4)
    ax.set_ylim(-0.75, len(MODELS) - 0.25)
    ax.annotate("chance = 0.500", xy=(0.5, -0.62), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0.45, 0.93)
    ax.set_yticks(y)
    ax.set_yticklabels([])

    fig.legend(handles=[Patch(facecolor=CLASSICAL, label="classical DNN"),
                        Patch(facecolor=HYBRID, label="hybrid QML (BasicEntangler)")],
               frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The team's original binary task: why accuracy cannot rank these models\n"
                 "2 577 compounds, random stratified split, \"active\" = IC50 < 10 µM → the test set is "
                 "91.7 % active",
                 color=INK, fontsize=12, x=0.004, ha="left", y=1.05)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / "fig4_prior_binary_accuracy.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    rows = ["# the team's ORIGINAL binary task -- not comparable with figures 1-3",
            f"# test set {N_TEST} compounds, {N_ACTIVE} active, trivial accuracy {TRIVIAL:.4f}",
            "model,accuracy,delta_vs_trivial,compounds_vs_trivial,balanced_accuracy,source"]
    src = ["saved notebook output", "teammate-reported", "saved notebook output"]
    for (lab, acc, bal, _), s in zip(MODELS, src):
        rows.append(f"\"{lab.replace(chr(10), ' ')}\",{acc:.4f},{acc - TRIVIAL:+.4f},"
                    f"{(acc - TRIVIAL) * N_TEST:+.1f},{bal if bal else 'NOT_REPORTED'},{s}")
    (FIGURES / "fig4_prior_binary_accuracy.csv").write_text("\n".join(rows) + "\n")
    print("wrote figures/fig4_prior_binary_accuracy.png and .csv")
    print(f"trivial baseline {TRIVIAL:.4f}; DNN {0.9251 - TRIVIAL:+.4f}, "
          f"2-rep {0.9200 - TRIVIAL:+.4f}, 4-rep {0.8863 - TRIVIAL:+.4f}")


if __name__ == "__main__":
    main()
