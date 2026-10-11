# Audit of the original notebooks

What we found when we read and re-measured the code on branch `main` (commits `1816ec6`, `a0d7240`).
Nothing in `classical/`, `quantum/`, `Dataset/`, `Test set/` or `Results_Presentation.pptx` was modified.

## 1. The 77–80 % accuracy plateau is an accuracy ceiling, not a training failure

This is the headline finding of the audit, and it is checkable in one command against the team's own
saved predictions (`Test set/pennylane_hybrid3_test_set_with_results.csv`, 387 rows):

| quantity | value |
|---|---|
| test-set class balance | active 303, intermediate 52, inactive 32 |
| **majority-class rate (always predict "active")** | **78.29 %** |
| the swept configurations' reported accuracy | 73.90 % – 80.36 % |
| best configuration's margin over always-guessing | **+2.07 points** |
| that same run's balanced accuracy / macro-F1 | 62.3 % / 59.0 % |

Every configuration in the sweep — 4q/2, 4q/4, 4q/8, 8q/1, 8q/2, 8q/4, 16q/1 reps, basic and strong
entangler — lands within a couple of points of the rate you get for free by predicting the majority class.
Accuracy on this split has almost no room to move, so a sweep scored on accuracy cannot distinguish
architectures and will look like it is "stagnating" no matter what the circuit does.

The models were **not** stuck. In the team's own numbers the discriminative metrics moved substantially
across the same sweep:

| configuration | accuracy | balanced accuracy | macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| 4 qubit, 2 reps (Qiskit) | 0.804 | 0.623 | 0.613 | 0.852 |
| 4 qubit, 4 reps | 0.749 | 0.645 | 0.595 | 0.847 |
| 4 qubit, 8 reps | 0.804 | 0.635 | 0.616 | 0.834 |
| 8 qubit, 1 rep | 0.791 | 0.629 | 0.615 | 0.834 |
| 8 qubit, 2 reps | 0.739 | 0.623 | 0.590 | 0.843 |
| 8 qubit, 4 reps | 0.778 | **0.671** | **0.626** | **0.860** |

The configuration with the *lowest* accuracy of the deeper runs (8q/4, 0.778) is the *best* model by
balanced accuracy, macro-F1 and AUC. Ranking by accuracy actively points the search the wrong way.

**What we changed as a result.** The extension scores every model on balanced accuracy, macro-F1, PR-AUC
and enrichment, and prints accuracy only next to the majority-class rate so the ceiling is always visible.

## 2. The statevector / default-simulator variance has a mundane cause

The report that "the normal simulator gives 74 % in one run and 82 % in another" is consistent with the
same accuracy ceiling plus seed noise: with 387 test compounds, one compound is 0.26 points, and the
confusion matrix above shows 52 active compounds sitting on the active/intermediate boundary. Moving a
handful of those across the decision boundary moves accuracy by several points while the ranking quality
barely changes. There is no need to invoke a difference between the simulators to explain an 8-point
accuracy swing; `lightning.qubit` and a shot-based device do differ, but the published runs were all
exact-expectation runs, so the swing is seed and threshold noise on a saturated metric.

## 3. Scientific issues found in the original pipeline

| # | issue | why it matters | what the extension does |
|---|---|---|---|
| 1 | **Mixed assay endpoints.** The ChEMBL pull keeps every IC50 row for CHEMBL243: 2 743 enzymatic rows, 286 unspecified-format, 89 cell-based and 31 organism-based, pooled into one label column. | A cell-based antiviral IC50 and an enzymatic inhibition IC50 are different measurements; pooling them puts incompatible numbers under one threshold. | Keeps enzymatic/biochemical rows only (`assay_type == B`, single-protein or assay format, cell-/antiviral-worded descriptions excluded). |
| 2 | **Duplicate compounds resolved by `drop_duplicates("canonical_smiles")`, keeping the first row.** | The retained value is whichever row ChEMBL returned first; 266 compounds have several measurements and 83 of those span more than a 10× range. | Aggregates by standardized InChIKey and takes the **median** pIC50; records the spread per compound. |
| 3 | **No structure standardization.** Salts, charge states and tautomer-equivalent SMILES are distinct strings, so the same parent can appear twice. | Inflates the compound count and can leak a parent across splits. | RDKit `LargestFragmentChooser` + `Uncharger`, then InChIKey deduplication. |
| 4 | **Random stratified split.** | Analogue series share a scaffold, so near-identical molecules land in train and test. Held-out performance then measures memorization of a series, not generalization to new chemistry. | Randomized Bemis–Murcko **scaffold split** (70/15/15, seed 2026) with an assertion that no scaffold crosses train↔test; IDs frozen in `data/v2/splits.csv`. |
| 5 | **ChEMBL validity flags ignored.** 92 rows carry `data_validity_comment = "Outside typical range"`. | These are the values ChEMBL itself flags as suspect. | Dropped. |
| 6 | **The binary task is 91.7 % positive.** | "Active" = IC50 < 10 µM makes nearly everything positive; 88.6 % accuracy is below the 91.7 % you get by answering "active" every time. | Primary endpoint moved to a label with usable balance (IC50 ≤ 10 nM, 39.7 % prevalence) and scored by enrichment. |
| 7 | **The test set guided architecture selection.** Nine configurations were compared by their test accuracy. | After nine looks, the test number is an optimistic estimate, not a held-out one. | New untouched split; depth and configuration chosen on validation only; test read once. |
| 8 | **`N_TRAIN = 800` subsamples the training set in the Qiskit notebook only.** | The Qiskit 4q/2 run and the PennyLane runs are not trained on the same data, so they are not comparable. | All compared models use the identical split and training set; any subsetting is applied to every model and recorded. |
| 9 | **Notebook `..._3_class_classifier.ipynb` cell 6 redefines `HybridMCQiskit` identically, and cell 14 refers to `HybridMC` and to a PennyLane checkpoint path**, so its last cell cannot run as saved. Cells 11–13 print output captured from the DNN notebook (`Best epoch: 52`, `DNN 3-class` row) rather than the Qiskit model's own metrics. | The 3-class Qiskit result table in that notebook is mislabelled: the displayed 0.804/0.623/0.613/0.852 row is the classical DNN's, not the hybrid's. | Reported as a reproduction blocker below; the extension's results are produced by a single script with saved run IDs. |

## 4. What we reused unchanged

- The **ChEMBL query and raw download** (`Dataset/csv_files/hiv1_protease_01_bioactivity_raw.csv`), target
  CHEMBL243, `standard_type = IC50`. All v2 data is derived from this file; nothing was re-downloaded.
- The **PaDEL PubChem fingerprints** (`Dataset/csv_files/descriptors_output.csv`, 881 bits) and
  `padel.sh`, matched to compounds by ChEMBL ID.
- The **three-class thresholds** (inactive ≥ 10 µM, active ≤ 1 µM) as a secondary task, unchanged.
- The **DNN architecture** (881 → 512 → 256 → 64 → out, BatchNorm, ReLU, Dropout 0.3) as the
  full-fingerprint classical baseline.
- The **hybrid head design** (Linear → tanh → π/2 angle scaling → circuit → ⟨Z⟩ → Linear), the
  U(0, 2π) circuit-weight initialization, class-weighted losses, AdamW + ReduceLROnPlateau, and
  early stopping on a validation metric.
- The **BasicEntangler circuit** (RX angle embedding, RX layers + CNOT ring) as model `be4`.

## 5. Reproduction status

| what | status |
|---|---|
| Re-ran the original notebooks end to end | **Not done.** They are Colab notebooks with `/content/...` paths, `!pip install`, `!wget`/`!unzip` of PaDEL, and `google.colab` imports; the 3-class Qiskit run alone took 72 minutes on Colab. |
| Inspected the saved outputs, checkpoints and logs | Done. All nine notebooks' configurations, metrics and saved weights were read; the numbers in §1 are from those files. |
| Verified the saved 3-class test predictions | Done. `Test set/pennylane_hybrid3_test_set_with_results.csv`: 387 rows, accuracy 0.7390, balanced accuracy 0.6231, macro-F1 0.5904, recomputed from the stored labels. |
| Reproduced the reported **80.36 %** | **Not attempted as a rerun.** It is recorded in the 4q/2 Qiskit and 4q/8 PennyLane notebooks' saved output and equals the DNN's accuracy to four decimals, which §3 item 9 explains. |
| Confirmed the quantum pathway is implemented as described | Done, by re-implementation plus numerical checks — see `qlink_screen/tests.py` (9/9 passing). |

The v2 results are therefore **not** a reproduction of the original runs. They are new measurements on a
new split with corrected data handling, and the two are kept apart everywhere in this repository.


## 6. The two sweep figures held back from the README

Four figures of the three-class sweep were contributed on 11 Oct. Two are in the README (figs 5 and 6).
Two are not, for reasons that are checkable against `qollab_combined/single_run.py`, which recomputes
every model's accuracy from its saved weights.

### `multiclass_benchmark_statevector_sim.png` — three bars disagree with the weights

| configuration | the figure | recomputed from the saved weights | the sampling-simulator figure |
|---|---|---|---|
| QML 4q, 2 rep | 80.36 % | **74.42 %** | 74.42 % |
| QML 4q, 4 rep | 80.36 % | **74.94 %** | 75.19 % |
| QML 4q, 8 rep | 74.94 % | **80.36 %** | 80.36 % |
| QML 8q, 1 rep | 79.07 % | 79.07 % | 79.07 % |
| QML 8q, 4 rep | 77.79 % | 77.78 % | 77.78 % |

The 4-qubit rows carry the classical DNN's 80.36 % where the 2-rep and 4-rep models belong, and the
4-rep and 8-rep values appear transposed. This is the same defect as §3 item 9: the notebooks printed a
stale `res`/`best_epoch` from the DNN run, and those printed numbers were charted. The sampling-simulator
figure, built by executing the weights, has the right values — which is why that one is in the README.

### `multiclass_benchmark_ION_realdevice_sim.png` — two problems

1. **The title says "Quantum Hardware Backend (IonQ forte Enterprise1 36q)".** Every run recorded in
   this repository reports `backend.name == "ionq_simulator"`, a noise model, and IonQ hardware consumes
   credits while the remote simulators are free. Unless a hardware job id can be produced, this must be
   labelled as a simulator. Claiming a hardware run we cannot evidence is the one error in this project
   that would be fatal with judges.
2. **The accuracies are 0 % or 100 % because each is measured on one molecule.** `BACKEND_BATCH = 1`, so
   every bar is a single prediction. Plotted on a 0–120 % axis it reads as some models being perfect and
   others failing completely; it is one coin flip each. The informative quantity from that run is the one
   the console prints — mean |ΔZ| against the exact values, which is what shows the noise damping.

Both figures also use two y-axes (accuracy left, epoch or deviation right). Two scales on one plot invite
a reader to see a relationship between the series that the arbitrary alignment of the axes invented.

**To use them:** regenerate the statevector figure from recomputed accuracies, retitle the IonQ figure as
a noise-model simulator, and report mean |ΔZ| rather than single-molecule accuracy. The numbers needed
are printed by `demo="hybrids"` on any backend.
