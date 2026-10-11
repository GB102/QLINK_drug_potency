"""Q-Link Screen -- compound prioritization for HIV-1 protease with a messenger-qubit circuit.

What this run does
  1. explains the screening problem and the assay budget K
  2. ranks a held-out pool of real ChEMBL compounds with the trained Q-LINK hybrid (exact expectations)
  3. compares that ranking with a classical random forest and with random picking
  4. re-runs a small selectable batch of compounds on the backend you picked in the Select QPU dialog,
     with the shot count you set, and shows the sampled expectations next to the exact ones

Everything printed under "live backend run" is executed during this run. Nothing is cached.
Trained weights are loaded from model_qlink.json; the training command is in the project README.

This effort is supported by Qollab & IonQ.
"""
import json
import math

from qlink_circuit import (build_from_spec, encode_angles, head, measured_circuit,
                           z_from_counts)

MODEL_FILE = "model_qlink.json"
POOL_FILE = "pool.json"
CLASSICAL_FILE = "classical_scores.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def exact_z(bound_circuit, n_data):
    """<Z_i> on each data qubit from the exact statevector (no shot noise)."""
    from qiskit.quantum_info import Statevector
    probs = Statevector(bound_circuit).probabilities_dict()
    z = [0.0] * n_data
    for bits, p in probs.items():
        for i in range(n_data):
            z[i] += p * (1 - 2 * int(bits[len(bits) - 1 - i]))
    return z


def enrichment(ranked_labels, n_actives, n_pool, k):
    hits = sum(ranked_labels[:k])
    base = n_actives / n_pool
    return hits, hits / k, (hits / k) / base if base else float("nan")


def rank_by(scores):
    """Descending by score, ties broken by pool order (identical rule for every model)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def ef_table(name, scores, labels, ks):
    n, n_act = len(labels), sum(labels)
    order = rank_by(scores)
    lab = [labels[i] for i in order]
    rows = []
    for k in ks:
        hits, prec, ef = enrichment(lab, n_act, n, k)
        rows.append((name, k, hits, prec, ef, hits / n_act))
    return rows


def print_rows(rows):
    w = max(len(r[0]) for r in rows) + 2
    print(f"  {'model':<{w}}{'K':>4}{'hits':>6}{'prec@K':>9}{'EF@K':>7}{'recall':>8}")
    for name, k, hits, prec, ef, rec in rows:
        print(f"  {name:<{w}}{k:>4}{hits:>6}{prec:>9.3f}{ef:>7.2f}{rec:>8.3f}")


def main(shots: int = 1024, backend_batch: int = 8, budget_percent: int = 10):
    spec = load_json(MODEL_FILE)
    pool = load_json(POOL_FILE)
    classical = load_json(CLASSICAL_FILE)
    compounds = pool["compounds"]
    n = len(compounds)
    labels = [c["potent"] for c in compounds]
    n_act = sum(labels)
    k = max(1, round(budget_percent / 100 * n))

    print("=" * 78)
    print("Q-Link Screen: can a messenger qubit improve compound prioritization?")
    print("=" * 78)
    print(f"Target      : {pool['target']}")
    print("Why it matters: HIV-1 protease cleaves the viral Gag-Pol polyprotein into the proteins a new")
    print("              virion needs. Block it and the virus still assembles but stays non-infectious,")
    print("              which is why protease inhibitors are a pillar of antiretroviral therapy.")
    print(f"Screening   : a wet-lab team can only assay a few compounds. We rank the pool and assay the")
    print(f"              top {budget_percent}% -- K = {k} of {n} compounds.")
    print(f"Pool        : {pool['split']}")
    print(f"Label       : {pool['label']}")
    print(f"Actives     : {n_act} of {n} compounds ({100 * n_act / n:.1f}%), so random picking recovers")
    print(f"              about {n_act / n * k:.1f} of them in a batch of {k} (EF = 1.0 by definition).")
    print()
    print(f"Model       : {spec['model']} ({spec['n_blocks']} blocks, {spec['n_qubits']} qubits ="
          f" {spec['n_data_qubits']} data + {spec['n_qubits'] - spec['n_data_qubits']} messenger)")
    print(f"              trained on the train split only; run id {spec['run_id']}")
    res = spec.get("resources", {}).get("ionq_native", {})
    if res:
        print(f"              compiled for IonQ's native set: depth {res.get('depth')}, "
              f"{res.get('two_qubit_gates')} two-qubit gates")
    print()

    qc, xs, ws, = build_from_spec(spec)
    weights = spec["circuit_weights"]
    n_data = spec["n_data_qubits"]

    # ---------------- 1. exact ranking of the whole pool (local, no shot noise) ----------------
    print("-" * 78)
    print("Step 1  Rank the full pool with exact expectation values (statevector, computed now)")
    print("-" * 78)
    q_scores, q_z = [], []
    for c in compounds:
        angles = encode_angles(c["features"], spec["pre_w"], spec["pre_b"], spec["angle_scale"])
        bound = qc.assign_parameters({**{p: float(v) for p, v in zip(xs, angles)},
                                      **{p: float(v) for p, v in zip(ws, weights)}})
        z = exact_z(bound, n_data)
        q_z.append(z)
        q_scores.append(head(z, spec["post_w"], spec["post_b"])[0])

    c_scores = [classical["scores"].get(c["id"], 0.0) for c in compounds]
    rng_state = 12345
    rand_scores = []
    for _ in range(n):                                  # deterministic LCG, so the run is reproducible
        rng_state = (1103515245 * rng_state + 12345) % (2 ** 31)
        rand_scores.append(rng_state / 2 ** 31)

    ks = sorted({max(1, round(p / 100 * n)) for p in (5, budget_percent, 20)})
    rows = []
    rows += ef_table(f"Q-LINK hybrid ({spec['model']})", q_scores, labels, ks)
    rows += ef_table("classical: random forest, full FP", c_scores, labels, ks)
    rows += ef_table("random picking", rand_scores, labels, ks)
    print_rows(rows)
    print()

    # ---------------- 2. the shortlist the model would send to the assay ----------------
    print("-" * 78)
    print(f"Step 2  The shortlist: top {k} compounds by Q-LINK score, with their known assay labels")
    print("-" * 78)
    order = rank_by(q_scores)[:k]
    print(f"  {'#':>3}  {'ChEMBL id':<16}{'score':>8}{'pIC50':>8}  {'assay label':<12}")
    for rank, i in enumerate(order, 1):
        c = compounds[i]
        print(f"  {rank:>3}  {c['id']:<16}{q_scores[i]:>8.3f}{c['pIC50']:>8.2f}  "
              f"{'POTENT' if c['potent'] else 'not potent':<12}")
    hits = sum(labels[i] for i in order)
    print(f"  recovered {hits} of {n_act} potent compounds in {k} assay slots "
          f"(random picking would recover about {n_act / n * k:.1f})")
    print("  Scores are uncalibrated model outputs used for ranking, not probabilities of activity.")
    print()

    # ---------------- 3. live backend run ----------------
    print("-" * 78)
    print("Step 3  Live backend run: the same quantum inference path, executed now on your backend")
    print("-" * 78)
    name = getattr(backend, "name", None)                                        # noqa: F821
    name = name() if callable(name) else (name or type(backend).__name__)        # noqa: F821
    batch = max(1, min(backend_batch, n))
    picked = rank_by(q_scores)[:batch]
    circuits = [measured_circuit(qc, xs, ws,
                                 encode_angles(compounds[i]["features"], spec["pre_w"], spec["pre_b"],
                                               spec["angle_scale"]),
                                 weights, n_data) for i in picked]
    print(f"  backend        : {name}")
    print(f"  shots          : {shots} per circuit")
    print(f"  circuits       : {batch} (one per compound, submitted as a single job)")
    print(f"  results        : newly executed in this run, not cached")
    print("  running ...")
    job = backend.run(circuits, shots=shots)                                     # noqa: F821
    result = job.result()

    print(f"  {'ChEMBL id':<16}{'exact <Z0..Zn>':<34}{'sampled':>9}{'exact':>9}{'|diff|':>8}")
    max_diff = 0.0
    for j, i in enumerate(picked):
        counts = result.get_counts(j) if batch > 1 else result.get_counts()
        z_s, total = z_from_counts(counts, n_data)
        s_sampled = head(z_s, spec["post_w"], spec["post_b"])[0]
        s_exact = q_scores[i]
        max_diff = max(max_diff, abs(s_sampled - s_exact))
        zs = " ".join(f"{v:+.2f}" for v in q_z[i])
        print(f"  {compounds[i]['id']:<16}{zs:<34}{s_sampled:>9.3f}{s_exact:>9.3f}"
              f"{abs(s_sampled - s_exact):>8.3f}")
    print(f"  largest score difference from finite sampling: {max_diff:.3f} at {shots} shots")
    order_s = [compounds[i]['id'] for i in picked]
    print(f"  shot-based run reproduced the exact top-{batch} ordering: "
          f"{order_s == [compounds[i]['id'] for i in rank_by(q_scores)[:batch]]}")
    print()
    print("Reproduce the training and the full benchmark with the commands in the project README.")
    print("A 4+1 qubit circuit is classically simulable: this is an architecture comparison on a real")
    print("assay benchmark, not a demonstration of computational quantum advantage.")
    print("This effort is supported by Qollab & IonQ.")
