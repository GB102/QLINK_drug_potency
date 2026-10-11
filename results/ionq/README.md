# Live IonQ runs through Qollab

Console output from `qollab_combined/` executed on Qollab's **IonQ Forte 1 remote simulator**
(`backend.name == "ionq_simulator"`), 11 Oct 2026. These are real runs, not simulations of runs:
each was produced by pressing Run in the Qollab playground with that backend selected.

| file | demo | backend circuit | shots | wall clock |
|---|---|---|---|---|
| `qlink_demo_ionq.txt` | `demo="qlink"` | 1 circuit, 15 qubits (3 compounds packed) | 1024 | 7 s |
| `hybrids_part0_ionq.txt` | `demo="hybrids", part=0` | 1 circuit, 28 qubits (3 models packed) | 1024 | 682 s |

## What they show

**Device noise damps the expectation values, and the ranking survives it.** In the qlink run every
score moved *down* by ~0.104, about 3x the 1-sigma shot noise at 1024 shots (0.031) and all in the same
direction — the signature of a depolarizing noise model shrinking |<Z>| toward zero, not of finite
sampling. The top-3 ordering was unchanged. The hybrids run reproduces the same effect independently:
|<Z>| is smaller on the backend than exact for two of the three models, and all three backend
predictions still agree with the exact ones.

That is the practical argument for compound prioritization as a near-term quantum application: the noise
destroys the scores as calibrated quantities while leaving the decision intact. Screening needs only the
ordering.

**Cost scales with circuit width, steeply.** IonQ's noise-model simulators run one simulation per shot,
so the 28-qubit hybrids circuit took 682 s against the 15-qubit qlink circuit's 7 s at the same shot
count. Width, not shot count, is what makes these runs expensive.
