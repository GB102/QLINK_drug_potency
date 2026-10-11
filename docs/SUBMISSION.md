# Submission materials

Deadline: **11 Oct 2026, 19:00 ET / 23:00 UTC**. Theme tag: **Quantum Impact**.

---

## 1. Project page draft (paste into Qollab)

### Q-Link Screen — can a messenger qubit improve compound prioritization?

A screening team has a fixed number of assay slots. We rank candidate compounds against **HIV-1 protease**
(ChEMBL `CHEMBL243`) so the first batch contains more genuinely potent molecules, and we test whether a
recent circuit design — **Q-LINK**, which adds one *messenger* qubit to carry information across layers —
prioritizes better than a matched ordinary circuit.

**What it does.** Press Run, choose a backend, and the demo explains the target, ranks a held-out pool of
60 real compounds, shows the shortlist with its true assay labels, and then runs the same quantum
inference path live on the IonQ simulator you selected, printing sampled expectation values beside exact
ones. Nothing under the live-run heading is cached.

**What we found, including what didn't work.**

- The messenger gives a small, repeatable edge over a matched ordinary circuit: **+0.075 EF@10 %** at both
  depths tested — about one extra potent compound per 36 assays. A 5-qubit control with an *idle* extra
  qubit scores identically to the 4-qubit circuit, so the effect comes from the messenger *interactions*,
  not from the extra qubit. That control is the result we trust most.
- The edge doesn't generalize: Q-LINK's PR-AUC is **lower** than the ordinary circuit's. It sharpens the
  top of the list without being better overall. With 3 seeds, we do **not** claim it improves screening.
- **Our extension — the sparse messenger — works at depth 4 and fails at depth 8.** Coupling every second
  block gives EF 2.21 with **28 compiled two-qubit gates**, matching full Q-LINK's EF 2.21 at **88 gates**
  — a 3.1× saving at equal quality. At depth 8 the same trick loses 0.43 EF on every seed.
- **No quantum advantage.** A plain classical MLP on the identical features beats every quantum model
  (EF 2.38 vs 2.21); a random forest on full fingerprints leads on PR-AUC (0.782 vs 0.680). The
  limitation is feature compression, not the circuit.
- **No barren plateau at this scale,** so this benchmark can't test Q-LINK's headline claim. Gradients are
  healthy throughout; Q-LINK's don't attenuate with depth (×1.04 over 4→8 blocks) while ordinary circuits'
  do (×0.96) — the right direction, nowhere near the paper's 8–10 qubit, 133–230 block regime.
- **512 shots are enough:** the top-36 selection overlaps the exact one by 92–97 %.

**Rigour.** Scaffold split with frozen IDs; enzymatic assays only; median pIC50 per standardized compound;
all preprocessing fit on train only; endpoint, budget K and metrics declared before any test evaluation;
test split read once; 9/9 circuit-correctness checks against Qiskit's own `Statevector` and
`StatevectorEstimator`, including a finite-difference gradient check.

**Honest limits.** 4+1 qubits is classically simulable — this is an architecture comparison on a real
assay benchmark, not computational quantum advantage. Three seeds, one target, one split. Retrospective
ChEMBL data: no clinical claim, no discovered drug, and model scores are ranking scores, not
probabilities.

**Built on our teammate's work.** The ChEMBL pull, PaDEL fingerprints, DNN and BasicEntangler hybrids came
first; this build added the Q-LINK implementation, the corrected dataset and scaffold split, the matched
controls, the screening metrics, the gradient and cost analysis, and this demo. Full attribution in
`docs/CHANGELOG.md`.

Repository: https://github.com/JuL-sezaR/QLINK_drug_potency · MIT licensed · Theme: Quantum Impact

*This effort is supported by Qollab & IonQ.*

---

## 2. Demo video script (2:30, ~370 words)

> Record the Qollab run live. Timings assume you let the terminal output scroll while you narrate.

**[0:00–0:20] The problem.**
"HIV-1 protease cuts the virus's own polyprotein into working pieces. Block it and the virus still
assembles, but it's not infectious — that's why protease inhibitors are a pillar of HIV therapy.
Suppose you can only afford to assay thirty-six compounds out of three hundred and sixty. Which ones?"

**[0:20–0:45] The question.**
"We took Q-LINK, a 2026 circuit design that adds one *messenger* qubit to carry information across layers,
and asked whether it picks a better batch than an ordinary variational circuit — and whether we can halve
the messenger's gate cost by coupling it only every second block. That second one is our own extension."

**[0:45–1:20] The demo running.** *(screen: press Run, pick IonQ simulator, show the dialog)*
"Everything runs in the browser on Qollab, on IonQ's simulator. The pool is held out by molecular
scaffold, so the test compounds are chemically distinct from anything the model trained on. Here's the
shortlist, with the real assay labels revealed — and here the same circuit executing live on the backend,
with sampled expectation values next to exact ones."

**[1:20–2:00] The results, good and bad.** *(screen: figures 1 and 3)*
"The messenger does help a little — about one extra potent compound per thirty-six assays, at both depths
we tested. And the control that matters: a five-qubit circuit with an *idle* extra qubit scores
identically to the four-qubit one, so the gain is from the messenger's interactions, not the extra qubit.
Our sparse version is the better story on cost — equal quality at twenty-eight two-qubit gates instead of
eighty-eight. But it falls apart at depth eight, on every seed."

**[2:00–2:30] What we won't claim.**
"We won't oversell it. A plain classical neural network on the same features beats every quantum model
here, and a random forest on full fingerprints beats them all on ranking quality. Four-plus-one qubits is
classically simulable, there's no barren plateau at this scale, and three seeds is three seeds. What we
have is a careful architectural comparison on a real assay benchmark, with the controls that make it
believable — and a cost result we'd want to test at depth. Supported by Qollab and IonQ."

---

## 3. Judging-criteria map

| criterion | evidence |
|---|---|
| **Novelty and Creativity** | First application we're aware of transferring Q-LINK from simulated ground-state preparation to molecular screening; the sparse-messenger variant is our own proposal, tested and reported with its failure case. The idle-qubit control isolates the messenger's contribution. |
| **Technical Execution** | Circuits built in Qiskit and verified against `Statevector`/`StatevectorEstimator` plus a finite-difference gradient check (9/9 in `qlink_screen/tests.py`). The training engine executes the same Qiskit circuit object the demo submits. Scaffold split with frozen IDs, train-only preprocessing, declared protocol, single test pass. Demo runs on Qollab's IonQ simulator with one job per run. |
| **Impact Potential** | Measured recovery of potent compounds under a fixed assay budget, with strong classical baselines kept in the comparison — including the ones that beat us. The transferable result is the quality-versus-gate-cost frontier, which is what decides whether a circuit is worth running on hardware. |
| **Documentation Quality** | README with results and limits up front; `AUDIT.md` (what was wrong in the originals and why, with the accuracy-ceiling analysis); `PREREGISTRATION.md` (protocol fixed before training); `CHANGELOG.md` (attribution and build-window disclosure); every figure shipped with its CSV; every number traceable to a saved run. |

---

## 4. Before you submit — checklist

- [ ] **Push the branch and make the repository public**, no sign-in, MIT licensed, no secrets. Replace
      https://github.com/JuL-sezaR/QLINK_drug_potency above.
- [ ] **Upload `qollab_combined/` as the Qollab project** (15 files, 10 MB, largest file 1.42 MB).
      It holds both demos behind one `main.py`: `demo = "qlink"` (the main demo) and
      `demo = "hybrids"` (the team's seven BasicEntangler models on the three-class task).
      One demo per run — IonQ accepts one job per run, so running both would submit two.
      `qollab/` (Q-Link Screen only) and `qollab/ramqolab/` (the sweep only) remain as fallbacks.
- [ ] **Press Run on an IonQ simulator and confirm it completes.** This is required and has not been done
      from this machine — see "Not verified" below.
- [ ] **Publish the project under MIT on Qollab** before 19:00 ET.
- [ ] **Record and upload the video** (2–3 min, public link, no sign-in).
- [ ] **Tag the theme: Quantum Impact.**
- [ ] **Disclose on the project page** which parts pre-existed, per `docs/CHANGELOG.md`.
- [ ] Captain submits from the registration page.

## 5. Not verified from here

**The Qollab/IonQ run has not been executed.** The platform needs a signed-in browser session, so
`backend` could not be exercised from this machine. What *was* verified locally:

- the demo runs end to end against a pre-created Qiskit `BasicSimulator` with the real weights and pool,
  producing the output above, with all circuits in a **single** `backend.run(list, shots=...)` call, which
  is what IonQ's one-job-per-run limit requires;
- `main.py` + `qlink_circuit.py` import nothing beyond Qiskit, `math` and `json` — no torch, no NumPy, no
  scikit-learn — so the Pyodide runtime can load them;
- data files are opened through plain string literals, as the runtime requires;
- `main()` takes only `int`/`bool` parameters, so the Run-settings form will render.

The remaining step is a human pressing Run with an IonQ backend selected. If it fails, the console
traceback plus Qollab's Error reference will say why; the most likely issue is multi-circuit job support,
in which case set `backend_batch=1`.
