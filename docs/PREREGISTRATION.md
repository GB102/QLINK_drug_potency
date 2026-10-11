# Pre-declared protocol (written before any v2 model was trained or any v2 test metric was computed)

Committed on branch `qlink-screen`, 11 Oct 2026, before running `qlink_screen.train`.

## Data
- Curated v2 set from the team's ChEMBL CHEMBL243 IC50 pull: enzymatic assays only, exact (`=`) nM values,
  ChEMBL validity-flagged rows removed, RDKit-standardized parents, median pIC50 per InChIKey.
- Randomized Bemis–Murcko scaffold split 70/15/15 (seed 2026); IDs frozen in `data/v2/splits.csv`.
- Test pool: 360 compounds (133 potent). The test split is touched once, by `qlink_screen.evaluate --final`.

## Endpoints
- **Primary screening label:** potent = median IC50 ≤ 10 nM (pIC50 ≥ 8). Rationale: 80 % of the enzymatic
  set already meets the team's 1 µM "active" bar, so ranking for ≤ 1 µM caps EF@10 % at ≈ 1.26 and cannot
  separate models. Fixed from the label distribution only, not from model results.
- **Primary metric:** EF@10 % of the held-out test pool (K = 36 of 360), with precision@K and recall@K.
  Ties are broken by stable sort order of the test-pool index (deterministic, identical for all models).
- **Secondary:** PR-AUC (potent vs rest), ROC-AUC, EF@5 %, EF@20 %, and the full EF-vs-K curve.
- **Secondary task (team's original):** 3-class (inactive ≥ 10 µM / intermediate / active ≤ 1 µM); balanced
  accuracy and macro-F1. Accuracy is reported only next to the majority-class rate.

## Models (compact features = train-only VarianceThreshold → PCA(16) → StandardScaler)
| id | description |
|---|---|
| `random` | random ranking (expected EF = 1) |
| `rf_fp` | random forest on full ECFP4-2048 + PubChem-881 fingerprints, tuned on validation PR-AUC |
| `dnn_fp` | team's DNN architecture (881→512→256→64→out) on variance-filtered PubChem fingerprints |
| `mlp` | MLP 16→64→out on compact features |
| `mlp_bneck` | classical 4-wide bottleneck 16→4→tanh→4→tanh→out (circuit replaced by a classical layer) |
| `be4` | team's quantum circuit family: RX angle encoding + BasicEntangler blocks (RX + CNOT ring), 4 qubits |
| `van4` | ordinary VQC with the Q-LINK paper's operation block (RZ·RY·RX + CZ chain), 4 qubits |
| `van5` | same block on 5 qubits; extra qubit starts in \|0⟩, receives no feature |
| `qlink` | 4 data + 1 messenger, Q-LINK (Fixed): H on messenger, RXX(π/4) collection before and CNOT distribution after every block |
| `qlink_sparse` | as `qlink`, messenger interactions only in every second block (blocks 0, 2, 4, …) |

All hybrid models: Linear(16→4) → (π/2)·tanh → circuit → ⟨Z⟩ on the 4 data qubits → Linear(4→out).
Circuit weights initialized U(0, 2π) (team's convention) for every circuit. Same optimizer (AdamW, lr 0.01,
wd 1e-4), batch 64, max 80 epochs, early stopping patience 15 on validation PR-AUC (3-class: macro-F1),
identical minibatch order for a given seed. Depths 4 and 8 blocks. Seeds 0, 1, 2.

## Selection rule
- Per model family, choose depth by mean validation PR-AUC over seeds; checkpoint = best validation epoch.
- Report test metrics for every seed of the chosen configuration: mean ± sd, plus paired per-seed
  differences against `van4` (does the messenger help?) and against `qlink` (does sparsening hurt?).
- A negative result is reported as found. No composite score is used to merge quality and cost.

## Resource accounting
Parameters, qubits, compiled depth and two-qubit gate count after `transpile(..., basis_gates=["rz","ry","rx","rxx","cx","h"])`
and for the IonQ native set, optimizer steps, circuit evaluations, wall-clock seconds on the recorded CPU.
Statevector training cost is reported separately from finite-shot inference cost (512 and 2048 shots).

## What this cannot show
4–5 qubits is classically simulable; any benefit found here is an architectural result on this benchmark,
not computational quantum advantage. Simulator epoch counts are never converted into hardware speedups.
