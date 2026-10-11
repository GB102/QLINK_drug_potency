# Q-Link Screen -- compound prioritization for HIV-1 protease, with a messenger-qubit circuit.
#
# Two demos share this project. Pick one with the `demo` field under Run settings:
#
#   demo = "qlink"    Rank a held-out pool of 60 real compounds with the Q-LINK hybrid (4 data qubits +
#                     1 messenger), compare against a random forest and random picking, show the
#                     shortlist with its true assay labels, then run the same quantum inference path
#                     live on your backend.
#
#   demo = "hybrids"  Score the team's seven earlier BasicEntangler hybrids (4, 8 and 16 qubits, 1-8
#                     reps) on the three-class task: exact expectation values for all seven, then a live
#                     backend run. On IonQ this needs two runs -- part = 0, then part = 1.
#
# One demo per run, because IonQ accepts one job per code run. Running both would submit two jobs and
# the second would be rejected.
#
# Trained weights are loaded from the project files; training is not run in the browser. Everything
# printed under a "live backend run" heading is executed in that run and is never cached.
#
# This effort is supported by Qollab & IonQ.

import qlink_demo
import single_run

TRIVIAL_3CLASS = 0.7829      # answering "active" for every molecule: 303 of the 387 test compounds


def short(res):
    """'hybrid_QML_3class_8qubit_2depth_basic_entangler.pt' -> '8 qubits, 2 reps'."""
    if res["n_qubits"] is None:
        return res["model_name"].replace("hybrid_QML_3class_", "").replace("_basic_entangler.pt", "")
    return f"{res['n_qubits']} qubits, {res['depth']} reps"


def run_hybrids(shots, part, molecules_per_model, start_index):
    results = single_run.main(backend,                      # noqa: F821  (Qollab pre-creates `backend`)
                              shots=shots, part=part,
                              backend_batch=molecules_per_model, start_index=start_index)
    print("\n" + "=" * 78)
    print("SUMMARY  -- exact accuracy on the full three-class test set")
    print("=" * 78)
    print(f"  {'model':<20}{'accuracy':>10}{'correct':>11}{'vs trivial':>12}")
    for res in sorted(results, key=lambda r: -(r["accuracy"] or 0)):
        if res["accuracy"] is None:
            print(f"  {short(res):<20}{'-':>10}{'-':>11}{'-':>12}   {res['status']}")
            continue
        fraction = f"{res['correct']}/{res['total']}"
        print(f"  {short(res):<20}{res['accuracy']:>10.4f}{fraction:>11}"
              f"{res['accuracy'] - TRIVIAL_3CLASS:>+12.4f}")

    ran = [r for r in results if r["backend_slice_total"]]
    if ran:
        print(f"\n  On the backend this run ({ran[0]['backend']}, {ran[0]['backend_shots']} shots):")
        for r in ran:
            last = r["backend_slice_start"] + r["backend_slice_total"] - 1
            print(f"    {short(r):<20}{r['backend_slice_correct']}/{r['backend_slice_total']} correct"
                  f"   on test molecules {r['backend_slice_start']}..{last}")

    print(f"\n  Accuracy note: {TRIVIAL_3CLASS:.4f} of this test set is \"active\", so a model that")
    print("  answers \"active\" for every molecule scores that. Read the column above against it, not")
    print("  against zero. Only one of the seven models beats it, and the best model by accuracy is")
    print("  not the best by balanced accuracy -- which is why the Q-Link Screen demo ranks compounds")
    print("  by enrichment instead. Run it with demo = \"qlink\".")
    return results


def main(demo: str = "qlink",
         shots: int = 1024,
         compounds: int = 3,
         part: int = 0,
         molecules_per_model: int = 1,
         start_index: int = 0):
    choice = (demo or "").strip().lower()
    if choice in ("qlink", "q-link", "screen", "1"):
        # `compounds` is how many are measured on the backend; `part` and the two index fields are unused.
        return qlink_demo.run(shots=shots, backend_batch=compounds, budget_percent=10)
    if choice in ("hybrids", "hybrid", "3class", "sweep", "2"):
        return run_hybrids(shots, part, molecules_per_model, start_index)
    print(f"Unknown demo {demo!r}. Set demo to \"qlink\" or \"hybrids\" under Run settings.")
    print("  qlink   -- rank compounds with the Q-LINK messenger-qubit circuit (the main demo)")
    print("  hybrids -- score the team's seven BasicEntangler hybrids on the three-class task")
    return None
