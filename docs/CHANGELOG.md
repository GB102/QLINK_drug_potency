# Change log — what came from where

Branch `qlink-screen`, built on `main` at commit `a0d7240`.

## Provenance of this repository

| component | origin | status |
|---|---|---|
| ChEMBL query and raw download for `CHEMBL243` | teammate (`Data_preprocessing_and_analysis_HIV_inhibitors.ipynb`) | **reused unchanged** — every v2 record derives from `Dataset/csv_files/hiv1_protease_01_bioactivity_raw.csv`; nothing was re-downloaded |
| PaDEL PubChem fingerprints (881 bits) + `padel.sh` | teammate | **reused unchanged** |
| 3-class thresholds (inactive ≥ 10 µM, active ≤ 1 µM) | teammate | **reused unchanged** as the secondary task |
| DNN architecture 881→512→256→64→out, BN/ReLU/Dropout 0.3 | teammate | **reused**, re-implemented as `models.DNNfp` |
| Hybrid head: Linear → tanh → ×π/2 → circuit → ⟨Z⟩ → Linear | teammate | **reused** — same design for every quantum model |
| Circuit-weight init U(0, 2π) | teammate | **reused** for every circuit family |
| Training policy: AdamW, ReduceLROnPlateau, class weights, early stopping on a validation metric | teammate | **reused** |
| BasicEntangler circuit (RX embedding, RX layers + CNOT ring) | teammate | **reused** as baseline `be4` |
| Data cleaning: assay-endpoint filter, standardization, median aggregation, validity flags | this extension | **corrected** — see AUDIT §3 items 1, 2, 3, 5 |
| Scaffold split with frozen IDs | this extension | **corrected** — replaces the random stratified split |
| Primary endpoint (IC50 ≤ 10 nM) and enrichment metrics | this extension | **added** — the previous binary task was 91.7 % positive |
| Q-LINK circuits (fixed, sparse, adaptive) | **method from arXiv:2604.11831**, implemented here | **added** |
| Paper's vanilla operation block; 5-qubit idle control | this extension | **added** |
| Differentiable statevector engine over Qiskit circuits | this extension | **added** |
| Gradient diagnostics, resource accounting, shot evaluation | this extension | **added** |
| Qollab/IonQ demo | this extension | **added** |

Nothing in `classical/`, `quantum/`, `Dataset/`, `Test set/` or `Results_Presentation.pptx` was modified,
moved or deleted. New outputs live in `data/v2/`, `results/`, `figures/`, `qollab/` and `docs/`.

## Reproducing each era's numbers

**Original results** — open the notebooks in `classical/` and `quantum/` in Colab. They are unchanged and
carry their own saved outputs. They were not re-executed here; see AUDIT §5.

**Extension results** — the command list in the README. Run IDs are
`results/runs/<task>__<model>__d<blocks>__s<seed>.json`, each with its own config, history, gradient
diagnostics, resource counts and wall-clock time.

The two sets are **not comparable**: different compound set, different split, different labels.

## Outside code, data and assistance

- **Q-LINK** method: Q-LINK, [arXiv:2604.11831](https://arxiv.org/abs/2604.11831). We read the paper and
  the authors' reference implementation at
  [AARC-lab/DCAS_2026_QLINK](https://github.com/AARC-lab/DCAS_2026_QLINK) to get the gate order, the
  control directions and the π/4 fixed coupling right. **That repository displays an MIT badge in its
  README but ships no LICENSE file**, so its permission status is unclear; we did not copy any of its
  code. Our circuits are written from scratch in Qiskit (it uses TensorCircuit) and the method is credited
  to its authors.
- **Data**: ChEMBL (EMBL-EBI), CC BY-SA 3.0. PaDEL-Descriptor via the `dataprofessor/bioinformatics`
  PaDEL bundle, as used by the teammate's original notebook.
- **Libraries**: Qiskit (Apache-2.0), Qiskit Aer (Apache-2.0), PyTorch (BSD-3), RDKit (BSD-3),
  scikit-learn (BSD-3), NumPy, pandas, Matplotlib.
- **AI assistance**: the extension in `qlink_screen/`, `qollab/` and `docs/` was written by Claude
  (Anthropic) working from the team's instructions and existing code, during the build window. All
  reported measurements come from executed runs saved in `results/`; no number in this repository was
  written by hand.

## Build-window disclosure — read before submitting

Qollab's rules say existing code is acceptable as a starting point **if disclosed**, and that judging is
based on what is added during the build window (9 Oct 19:00 ET → 11 Oct 19:00 ET).

- Commits `1816ec6` and `a0d7240` (the teammate's notebooks, dataset, checkpoints and slides) are dated
  **11 Oct 2026** — inside the window — but the notebook outputs are timestamped **10 Oct 2026, 23:26 UTC
  onward**, which is also inside the window (kickoff was 9 Oct 23:00 UTC). On the evidence in the
  repository, the prior work was done during the event.
- Everything on branch `qlink-screen` was created during the window.
- **Action for the team:** state plainly on the project page which parts pre-existed this agent's work —
  the ChEMBL pull, the fingerprints, the DNN and the BasicEntangler hybrids — and that the Q-LINK
  implementation, the corrected dataset, the controls, the metrics and the demo were added during the
  build. If any of the teammate's work actually started before 9 Oct 23:00 UTC, disclose that date; do
  not rely on the commit timestamps alone.
