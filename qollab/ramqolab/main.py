# Three-class HIV-1 protease classification with hybrid quantum-classical models.
#
# Evaluates seven BasicEntangler hybrids (4/8/16 qubits, 1-8 reps) on the held-out test set:
# exact expectation values locally for every model, then a live backend run.
#
# Why everything goes through single_run: IonQ accepts ONE job per code run, and each
# hybrid_QML3class_*_model.py calls backend.run() itself. Calling several of them in one run therefore
# submits several jobs and IonQ rejects every one after the first -- which is why this file used to fail
# on IonQ while single_run.py worked. single_run covers all seven models in a single job by packing them
# side by side into one circuit, so those per-model modules are not needed here.
#
# A backend run shows `molecules_per_model` molecules per model. The models do not all fit in one circuit,
# so the run prints a plan split into parts; run each PART in turn to cover every model.

import single_run

TRIVIAL = 0.7829          # answering "active" for every molecule: 303 of the 387 test compounds


def short(res):
    """'hybrid_QML_3class_8qubit_2depth_basic_entangler.pt' -> '8 qubits, 2 reps'."""
    if res["n_qubits"] is None:
        return res["model_name"].replace("hybrid_QML_3class_", "").replace("_basic_entangler.pt", "")
    return f"{res['n_qubits']} qubits, {res['depth']} reps"


def main(shots: int = 1024,
         part: int = 0,
         molecules_per_model: int = 1,
         start_index: int = 0):
    results = single_run.main(backend,                      # noqa: F821  (Qollab pre-creates `backend`)
                              shots=shots,
                              part=part,
                              backend_batch=molecules_per_model,
                              start_index=start_index)

    print("\n====================================")
    print("SUMMARY")
    print("====================================")
    print(f"{'model':<20}{'accuracy':>10}{'correct':>11}{'vs trivial':>12}")
    for res in sorted(results, key=lambda r: -(r["accuracy"] or 0)):
        if res["accuracy"] is None:
            print(f"{short(res):<20}{'-':>10}{'-':>11}{'-':>12}   {res['status']}")
            continue
        fraction = f"{res['correct']}/{res['total']}"
        delta = f"{res['accuracy'] - TRIVIAL:+.4f}"
        print(f"{short(res):<20}{res['accuracy']:>10.4f}{fraction:>11}{delta:>12}")

    ran = [r for r in results if r["backend_slice_total"]]
    if ran:
        print(f"\nOn the backend this run ({ran[0]['backend']}, {ran[0]['backend_shots']} shots):")
        for r in ran:
            last = r["backend_slice_start"] + r["backend_slice_total"] - 1
            print(f"  {short(r):<20}{r['backend_slice_correct']}/{r['backend_slice_total']} correct"
                  f"   on test molecules {r['backend_slice_start']}..{last}")

    print(f"\nAccuracy note: {TRIVIAL:.4f} of the test set is \"active\", so a model that answers")
    print("\"active\" for every molecule scores that. Read the column above against it, not against zero;")
    print("balanced accuracy and macro-F1 separate these models, accuracy does not.")
    return results
