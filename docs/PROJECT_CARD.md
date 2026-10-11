# Q-Link Screen

*Does a messenger qubit pick a better drug-screening shortlist? We test Q-LINK on held-out HIV-1 protease
assay data against matched classical and quantum controls, and report where it wins and where it doesn't.*

**Theme: Quantum Impact** · Qiskit · runs on IonQ simulators through Qollab

---

## 1. What is it?

A compound-prioritization benchmark with an honest scorecard.

Given 360 real molecules tested against **HIV-1 protease** (ChEMBL target `CHEMBL243`), we rank them so the
first batch sent to the lab contains as many genuinely potent compounds as possible. We score a hybrid
quantum classifier built on **Q-LINK** — a 2026 circuit design that adds a single *messenger* qubit to
carry information across layers — against a matched ordinary quantum circuit, three classical baselines,
and random picking.

We also propose and test our own variant: a **sparse messenger** that couples only every second block, to
see whether the messenger's benefit survives at a fraction of the entangling-gate cost.

Press **Run** and the demo ranks a held-out pool, shows the shortlist with its real assay labels, and then
executes the same quantum inference path **live on the IonQ simulator you select**.

## 2. Why does it matter, and who is it for?

HIV-1 protease cuts the virus's own Gag-Pol polyprotein into the parts a new virion needs. Block it and
the virus still assembles but stays non-infectious — which is why protease inhibitors are a pillar of
antiretroviral therapy.

**For a screening chemist**, the bottleneck is never ideas, it's assay slots. If you can afford 36 assays
out of 360 candidates, picking at random gets you about 13 potent compounds. Our models get **29 to 32**.
That ratio — the *enrichment factor* — is the number that decides whether a model is worth using, and it
is the number we optimize and report. Not accuracy.

**For a quantum practitioner**, the useful finding is about *what kind of task survives a real device*.
Our IonQ run (below) shows a noise model that visibly corrupts the model's output values while leaving the
**ranking intact**. Screening only needs the ordering. That makes compound prioritization a far more
noise-tolerant target for near-term hardware than any task needing calibrated probabilities — and that
argument is measured here, not asserted.

## 3. How does the quantum part work?

Sixteen compressed molecular-fingerprint features enter a linear layer that outputs four angles. Those
angles are RX-encoded onto four **data qubits**. A fifth **messenger qubit** starts in |0⟩, gets an H, and
then, following the Q-LINK authors' reference implementation gate for gate:

```
RXX(π/4) from every data qubit to the messenger     ← collection
per block:  RZ, RY, RX on each data qubit;  CZ chain
            CNOT(messenger → each data qubit)       ← distribution
            RXX(π/4) collection again, unless last block
```

The messenger is **never measured or reset** — it stays coherent to the end and is excluded from the
observables, so it acts as a persistent side channel rather than a mid-circuit readout. We read ⟨Z⟩ on the
four data qubits and feed them to a small linear head. The sparse variant does collection/distribution
only around blocks 0, 2, 4, … and keeps every other gate identical.

**Correctness is checked, not assumed.** Qiskit's `RXXGate` is `exp(-iθ/2·X⊗X)`, which we verified matches
the TensorCircuit convention the reference uses, so π/4 carries over. `qlink_screen/tests.py` runs **9
checks** — amplitudes and ⟨Z⟩ against Qiskit's own `Statevector` and `StatevectorEstimator`, batching,
input and weight dependence, a finite-difference gradient check, and the structural invariants of each
circuit. All 9 pass. Training differentiates *the same Qiskit circuit object* the demo submits to IonQ.

**One engineering detail worth stealing:** IonQ accepts exactly one circuit per job. So when the demo
measures several compounds, it packs them into a **single circuit**, each compound on its own block of 5
qubits. The blocks are unentangled, so one shot samples every compound at once and per-compound statistics
come from marginalising the bitstring. On local simulators the demo sends separate circuits instead —
Qiskit's `BasicSimulator` simulates once *per shot*, so a wide packed circuit would cost ~1000× more there.

## 4. How do I run it?

**On Qollab:** press **Run**, pick a backend in the Select QPU dialog, and choose a demo under Run
settings. Choose an **IonQ** remote simulator for the full effect.

**This project holds two runnable demos**, selected by the `demo` field. One per run — IonQ accepts one
job per code run, so running both at once would submit two jobs and the second would be rejected.

### `demo = "qlink"` — the main demo (about 7 s on IonQ)

Ranks a held-out pool of 60 real compounds with the Q-LINK hybrid, compares it against a random forest
and random picking, shows the shortlist with its true assay labels, then runs the same quantum inference
path live on your backend and prints sampled expectation values beside exact ones.

| parameter | default | what it does |
|---|---|---|
| `shots` | 1024 | shots per run. IonQ's noise-model simulators simulate once per shot. |
| `compounds` | 3 | compounds measured on the backend. Sets the packed circuit's width (3 × 5 = 15 qubits, inside Forte 1's 29). |
| `budget_percent` | 10 | the assay budget K, as a percentage of the pool. |

### `demo = "hybrids"` — the companion sweep (about 11 min on IonQ at 28 qubits)

Scores the team's seven earlier BasicEntangler hybrids (4, 8 and 16 qubits, 1–8 reps) on the three-class
task: exact expectation values for all seven, then a live backend run. The seven do not fit in one
29-qubit circuit, so on IonQ it runs in two parts.

| parameter | default | what it does |
|---|---|---|
| `shots` | 1024 | shots per run. |
| `part` | 0 | which group of models goes to the backend. Run `0`, then `1`, to cover all seven. |
| `molecules_per_model` | 1 | test molecules per model in the backend circuit. |
| `start_index` | 0 | first test molecule of the backend slice. |

Be aware of the cost: these runs are wide, and IonQ's noise-model simulators simulate once per **shot**,
so a 28-qubit circuit took 682 s against the qlink demo's 7 s at the same shot count. Width is what makes
a run expensive here, not the shot count.

Trained weights are loaded from the project files so the browser doesn't retrain — the page says so, and
everything printed under a live-run heading is executed in that run, never cached.

**Locally**, the full benchmark is `pip install -r requirements.txt` then ten commands listed in the
README — about 9 minutes end to end on 24 threads, including the training of all 45 runs.

---

## What we found

Held-out **scaffold split** (no shared molecular scaffold between train and test), 360 test compounds, 133
potent (IC50 ≤ 10 nM), budget K = 36. Endpoint, budget and metrics were fixed **before** any test
evaluation; the test split was read once. Three seeds.

| model | EF@10 % | potent found in 36 assays | PR-AUC | 2q gates |
|---|---|---|---|---|
| classical MLP, same features | **2.38** | 31.7 | 0.736 | – |
| random forest, full fingerprints | 2.28 | 30.3 | **0.782** | – |
| **Q-LINK (4+1)** | 2.21 | 29.3 | 0.680 | 88 |
| **Q-LINK sparse (4+1)** | 2.21 | 29.3 | 0.675 | **28** |
| ordinary VQC (4 qubits) | 2.08 | 27.7 | 0.704 | 12 |
| ordinary VQC (5 qubits, one idle) | 2.08 | 27.7 | 0.699 | 12 |
| random picking | 0.88 | 11.7 | 0.365 | 0 |

**The messenger helps slightly, and we can prove it's the messenger.** Q-LINK beats the matched ordinary
circuit by +0.075 EF at *both* depths tested. The control that makes this meaningful: a 5-qubit circuit
with an **idle** extra qubit scores *identically* to the 4-qubit one. So the gain comes from the messenger
*interactions*, not from a wider Hilbert space.

**But it doesn't generalize.** Q-LINK's PR-AUC is *lower* than the ordinary circuit's (0.680 vs 0.704) —
it sharpens the top of the list without being better overall. With three seeds, we do not claim Q-LINK
improves screening.

**Our sparse variant works at depth 4 and fails at depth 8.** Sparse at 4 blocks matches full Q-LINK at 8
blocks on enrichment using **28 compiled two-qubit gates instead of 88** — a 3.1× saving at equal quality.
At depth 8, the same trick loses 0.43 EF on *every* seed. A shallow circuit tolerates refreshing the
messenger half as often; a deeper one does not.

**No quantum advantage.** A plain classical MLP on the identical features beats every quantum model. The
limitation is feature compression, not the circuit — a classical 4-wide bottleneck scores 2.11, within
noise of the quantum circuits.

**No barren plateau at this scale,** so this benchmark cannot test Q-LINK's headline claim either way.
Gradients are healthy throughout (≈10⁻²). Q-LINK's don't attenuate with depth (×1.04 over 4→8 blocks)
while ordinary circuits' do (×0.96) — the right direction, but nowhere near the paper's 8–10 qubit,
133–230 block regime.

### The IonQ run: ranking survives noise that calibration doesn't

Running on `ionq_simulator` through Qollab, 15 qubits, 1024 shots, 9 s:

| compound | exact score | on IonQ | shift |
|---|---|---|---|
| CHEMBL269252 | 0.971 | 0.892 | −0.079 |
| CHEMBL52702 | 0.961 | 0.852 | −0.110 |
| CHEMBL360212 | 0.957 | 0.853 | −0.104 |

Every score moved **down**, by ~3× more than 1-sigma shot noise (0.031) would explain. That one-directional
shift is the device noise model damping ⟨Z⟩ toward zero, which drags every score toward the head's bias.
Finite sampling would scatter symmetrically; this does not.

The `hybrids` demo reproduces the effect independently on a 28-qubit circuit: |⟨Z⟩| is smaller on the
backend than exact for two of its three models, and all three backend predictions still match the exact
ones. Both run logs are saved in [`results/ionq/`](../results/ionq/).

**And the ranking was unchanged.** The compounds came back in the same order. The shortlist you'd send to
the bench is identical. That is the practical case for screening as a near-term quantum application: the
noise destroys the scores as probabilities while leaving the decision intact.

### Where's the accuracy?

Deliberately not the headline — and this is the single most useful thing we learned from auditing our
teammate's earlier sweep.

On their three-class task, the test set is **78.29 % "active"**, so answering "active" for everything
scores 78.29 %. Every configuration they tried — 4q/2, 4q/4, 4q/8, 8q/1, 8q/2, 8q/4, 16q/1, basic and
strong entangler — landed between **73.9 % and 80.4 %**, within a couple of points of free. The apparent
"plateau at 77–80 %" was the class-imbalance ceiling of the metric, not a training failure. Worse, it
pointed the search the wrong way: their 8q/4 run had the *lowest* accuracy of the deep runs (0.778) and
the *best* balanced accuracy (0.671), macro-F1 (0.626) and AUC (0.860).

For the record, on our balanced task (majority-class rate 0.631): Q-LINK 0.695, sparse 0.700, ordinary VQC
0.728, MLP 0.748, random forest 0.790 — and `precision@K = 0.815` is the number a chemist actually feels,
since it means 29 of 36 assays hit.

## Honest limits

- **4+1 qubits is classically simulable.** This is an architecture comparison on a real assay benchmark,
  not computational quantum advantage. Simulator epochs are never converted into hardware speedups.
- **Three seeds**, and EF moves in steps of 0.075 on this pool, so most paired differences sit inside
  noise. The two results consistent across every seed are the idle-qubit control and the depth-8 sparse
  regression.
- **No QPU run.** Training and the benchmark are exact statevector; the hardware-flavoured numbers come
  from IonQ's noise-model simulator and from Qiskit transpilation to IonQ's native gate set.
- **Retrospective study** on known ChEMBL results. No clinical claim, no discovered drug. Model scores are
  uncalibrated ranking scores, not probabilities of activity — doubly so after the noise result above.
- **One target, one split.** Nothing here is claimed to generalize beyond CHEMBL243.

## Credit

**Q-LINK** is not ours: Q-LINK, [arXiv:2604.11831](https://arxiv.org/abs/2604.11831), with a reference
implementation at [AARC-lab/DCAS_2026_QLINK](https://github.com/AARC-lab/DCAS_2026_QLINK). We read both to
implement the method faithfully in Qiskit; we copied no code (that repo shows an MIT badge but ships no
LICENSE file). The sparse-messenger variant, the screening benchmark and this analysis are ours.

The second demo, `demo = "hybrids"`, is our teammate's earlier sweep — seven BasicEntangler hybrids on
the three-class task — and it reaches the same conclusion from the other direction: only one of the seven
beats the 0.7829 majority-class rate, and the best model by accuracy is not the best by balanced
accuracy. That result is what moved this project off accuracy and onto enrichment.

Built on our teammate's prior work in the same repository — the ChEMBL pull, the PaDEL fingerprints, the
classical DNN and the BasicEntangler hybrids. This build added the Q-LINK implementation, a corrected
dataset and scaffold split, the matched controls, the screening metrics, the gradient and cost analysis,
and this demo. Full attribution and build-window disclosure: `docs/CHANGELOG.md`.

Data: ChEMBL (EMBL-EBI), CC BY-SA 3.0. Libraries: Qiskit, Qiskit Aer, PyTorch, RDKit, scikit-learn.
MIT licensed. Repository: https://github.com/JuL-sezaR/QLINK_drug_potency

*This effort is supported by Qollab & IonQ.*
