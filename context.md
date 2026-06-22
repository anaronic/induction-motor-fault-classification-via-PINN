# Project Context — Physics-Informed Motor Fault Classification (PINN)

> Resume file for the assistant. Read this first when reopening the project.

## What this project is
Physics-informed neural-network pipeline classifying induction-motor **bearing faults**
(healthy / inner_race / outer_race / rolling_element) on the **Paderborn University Bearing
Dataset**. Authors: Anaranyo Sarkar & Vanessa Shrie (3rd-year B.Tech EE, MNNIT Allahabad).
Personal GitHub: `anaronic` (push over HTTPS).

## Pipeline (end-to-end)
`src/pinn_motor_fault/`
- **paderborn.py** — download/extract official `.rar` archives, load `.mat`, infer label from
  bearing code (K00*=healthy, KI*=inner, KA*=outer, KB*=rolling), window signal (4096/stride
  2048), parse shaft freq from filename (`N##` → rpm/60).
- **features.py** — per-window z-norm, then 10 time-domain + 7 spectral + envelope-spectrum
  harmonic-energy features around BPFO=3.5848·fs, BPFI=5.4152·fs, BSF=2.3570·fs. Also builds
  `physics_targets` (softmax over characteristic-frequency energy).
- **model.py** — 1-hidden-layer NumPy NN (tanh + softmax). Loss = cross-entropy + λ·‖ŷ−y_physics‖²
  (λ≈0.20). `Standardizer` is **fit on train only** (line ~51), applied to test via `transform`
  (line ~98) → no scaling leakage.
- **train.py** — `stratified_split` (random), `stratified_group_split` (by `.mat` file, used in
  final experiment), `run_grouped_experiment`, synthetic smoke data.
- **results.py** — metrics, confusion matrix, feature importance, SVG figures, LaTeX tables.
- **cli.py** — `download`, `train`, `smoke`, `results`, `experiment`. Entry point
  `pinn-motor-fault`.

## Headline result (current)
- All **32 archives**, 2559 readable `.mat`, ~82k windows, grouped-by-file split, 200 epochs,
  32 windows/file → **86.38% held-out-file accuracy** (`reports/results_final/metrics.json`).

## Professor's concerns (addressed 2026-06-22)
He asked: (1) how is train/test split done, (2) how is scaling done, (3) suspected rolling-window
data leakage because the old 4-archive report showed ~99.5%.

**Findings (from `scripts/leakage_check.py`, runs same model under 3 splits):**
| Split | Full 32-archive test acc | Old 4-archive subset |
|---|---|---|
| Random window | 0.799 | 0.994 |
| Grouped by `.mat` file (leak-free) | 0.783 | **0.998** |
| Held-out whole bearing code (strictest) | **0.433** | impossible (0.25 chance) |

- **Rolling-window leakage ruled out:** random vs grouped differ <2 pts. Grouped (leak-free) split
  on the 4-archive subset *also* gives ~99.8% → the old 99% was **not** leakage and **not** better
  data; it was a trivially easy one-bearing-per-class task with no intra-class variability.
- **Real limitation = bearing-instance generalisation:** holding out whole bearing codes drops to
  ~0.43. Both random and grouped splits still let the same bearing code appear (different
  load/speed) in train and test.
- **Scaling:** per-window z-norm (self-contained) + feature standardiser fit on train only. Clean.

Reply drafted (concise email) at `reports/reply_to_professor.md`.

## Repo state / artifacts
- `scripts/leakage_check.py` — reproducible cross-check (args: `--codes`, `--max-windows-per-file`,
  `--epochs`). Run: `python scripts/leakage_check.py` (all codes) — needs `data/paderborn/extracted`.
- `reports/progress_report.tex` — full LaTeX report; added **Data Leakage Cross-Check** and
  **Data Scaling** sections.
- `reports/reply_to_professor.md` — short email to professor.
- Data/models are gitignored; full extracted dataset IS present locally at
  `data/paderborn/extracted` (32 code folders).
- Old reference: `Progress Report BTech Project (old).pdf` (root, untracked).

## Env / commands (verified)
- Python 3.14, numpy 2.4.6, scipy 1.17.1. Package installed editable.
- Tests: `python -m pytest -q` (1 test, passes).
- Reproduce final experiment:
  `pinn-motor-fault experiment --data-dir data\paderborn\extracted --epochs 200 --max-windows-per-file 32 --signal vibration --model models\paderborn_final_grouped_pinn.npz --output-dir reports\results_final`

## Next steps (proposed to professor)
1. Report a leakage-free **held-out bearing-ID / operating-condition** split openly (~0.43 honest).
2. Add classic baselines: SVM, Random Forest, plain MLP.
3. Physics-loss **ablation** (λ=0 vs λ>0) to show the physics term helps.
4. Optional: CNN/PINN hybrid; fuse current + vibration channels.

## User preferences (this user: @t-anasarkar / anaronic)
- **No** `Co-authored-by` trailer in commit messages.
- Pushes personal projects from personal GitHub account `anaronic` (HTTPS).
- Keep replies/email concise.
