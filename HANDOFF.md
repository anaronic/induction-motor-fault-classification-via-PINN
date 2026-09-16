# Handoff: Induction Motor Fault Classification (Physics-Informed, Paderborn Dataset)

## Project context

B.Tech project (MNNIT Allahabad, Electrical Engineering) classifying induction
motor bearing faults from the Paderborn University bearing dataset, using a
lightweight NumPy physics-informed neural network (PINN-style: cross-entropy
loss + a physics-consistency loss term based on bearing characteristic
fault frequencies). Work happens in two places:

- **Local repo** (this codebase) — edited in VS Code, has an AI coding
  assistant (you) available.
- **Kaggle notebooks** — where actual training runs happen, because the
  dataset (~17GB) can't be downloaded locally. The local repo gets zipped and
  re-uploaded to Kaggle as a dataset (`anaranyosarkar27/pinn-motor-fault-code-v2`
  as of the last upload) whenever it changes. There is no live sync — changes
  made locally must be manually re-zipped and re-uploaded to Kaggle before a
  Kaggle notebook will see them.

Repo root: `induction-motor-fault-classification-via-PINN/`
Package: `src/pinn_motor_fault/` (installed editable via `pip install -e .`)

## Package layout

- `paderborn.py` — downloads/loads Paderborn `.mat` files into windowed
  signal arrays. `load_paderborn_windows()` is the main entry point.
- `features.py` — `PhysicsFeatureExtractor` computes ~36 time/spectral/
  envelope-harmonic features per window, plus `physics_targets` (a 4-column
  soft target per window derived from bearing characteristic-frequency
  harmonic energies, used by the physics-consistency loss).
- `model.py` — `PhysicsInformedNN`: small NumPy feedforward net (tanh hidden
  layer, softmax output). `class_names` is a constructor param — output layer
  size is `len(class_names)`, so it supports both the original 4-class
  setup and the newer binary setup without code changes. `.fit()` runs an
  internal **25-combination hyperparameter grid search** (5 learning rates ×
  5 physics weights, each trained for the full epoch count) before final
  training — this is expensive and was bypassed today via `fast_train.py`.
  Has its own internal `Standardizer` (fit inside `.fit()`/`fast_fit()`).
- `fast_train.py` — **added today**. `fast_fit()` trains without the grid
  search (fixed hyperparameters, single pass) for time-constrained runs.
  `compute_class_weights()` factors out the inverse-frequency + focal-loss
  class-weighting formula so every model in a comparison uses the same
  scheme.
- `paper_features.py` — **added today, updated since**. `PaperFeatureSelector`
  implements a 4-class-adapted (now also binary-compatible) one-vs-rest
  version of Algorithm 1 from the reference paper (Li et al., EUSIPCO 2025):
  z-scores the averaged FFT magnitude spectrum per class vs. the rest, takes
  the max one-vs-rest z-score difference per frequency bin, and selects bins
  inside the physics-defined candidate set B where the difference is
  `>= tau` (default `tau=1.0`). This matches the paper's own threshold-based
  selection criterion (Step 5 of Algorithm 1) — an earlier top-K=20 version
  has been replaced. If `tau` selects zero features, `fit()` raises
  `RuntimeError` with the max observed difference, rather than silently
  falling back to arbitrary bins.
- `train.py` — original training entry points (`train_from_paderborn`,
  `train_from_windows`, `run_grouped_experiment`) using the full grid search.
  `stratified_group_split()` here is the standard train/test split used
  everywhere — splits by `.mat` file/source, not by window, to avoid leakage.
- `results.py` — `write_evaluation_artifacts()` generates metrics.json,
  confusion_matrix.csv/svg, feature_importance.csv/svg (weight-magnitude
  method — see below), class_metrics.csv, training_curves.svg,
  signal_examples.svg, metrics_table.tex.
- `cli.py` — `download` / `train` / `smoke` / `results` / `experiment`
  subcommands.

## Bugs found and fixed today (already applied in the local repo)

1. **`features.py`, `_time_features()`**: `entropy = -np.sum(x * np.log(x +
   1e-12))` — `x` can be negative (it's a mean-centered signal), producing
   NaN losses. Fixed to `p = np.abs(x) / (np.sum(np.abs(x)) + 1e-12);
   entropy = -np.sum(p * np.log(p + 1e-12))`.
2. Added `tqdm` progress bars in `paderborn.py` (file loading loop) and
   `features.py` (feature extraction loop) — cosmetic, no functional change.
3. `train.py`, `run_grouped_experiment()`: features were fed to the model
   unstandardized, which makes `results.py`'s weight-magnitude feature
   importance unreliable (features on wildly different raw scales — e.g.
   `spectral_centroid` in the thousands of Hz vs. harmonic-energy ratios in
   0–1 — get artificially small/large weights just to balance their
   contribution). Fixed by fitting a `Standardizer` on the train split only
   and transforming both splits before training/eval. **Note**: `model.py`
   was later found to already do its own internal standardization inside
   `.fit()`/`fast_fit()` (`self.standardizer`), so this train.py-level fix
   may now be partially redundant — not harmful, but worth reviewing whether
   both standardization layers are needed or if the outer one in
   `run_grouped_experiment` should be removed.

## Known bearing-geometry caveat (not yet fixed)

`BearingPhysics` in `features.py` hardcodes BPFO/BPFI/BSF ratios
(3.5848 / 5.4152 / 2.3570) that are the published values for an **SKF 6205**
bearing (commonly cited from the CWRU dataset). The Paderborn dataset's
actual bearing is a **type 6203** (per Lessmeier et al. 2016, the dataset's
own reference paper), which has different geometry and therefore different
true characteristic-frequency ratios. This means the physics-target /
harmonic-energy features may be computed around the wrong frequencies.
**Not yet corrected** — either find the actual 6203 geometry (ball count,
ball diameter, pitch diameter, contact angle) and recompute the ratios via
the standard BPFO/BPFI/BSF formulas, or document this as an explicit
limitation.

## Project scope change (mid-session, per professor feedback)

The project moved from 4-class (healthy / inner_race / outer_race /
rolling_element) to **binary classification** (healthy vs. faulty, with
inner/outer/rolling collapsed into "faulty"), specifically so results can be
compared against the reference paper's binary classifier. This is the
current, authoritative scope — **any earlier 4-class results/heatmaps/SHAP
outputs from earlier today are superseded and should not be mixed with
binary results.**

Binary conversion recipe (used in both binary notebooks below):
```python
labels = np.where(original_labels == "healthy", "healthy", "faulty")
binary_physics_targets = np.column_stack([
    batch.physics_targets[:, 0],                 # healthy column unchanged
    batch.physics_targets[:, 1:].sum(axis=1),    # sum of the 3 fault columns
])  # each row already sums to 1, no renormalization needed
```

## Current comparison design (3-way, binary)

| Notebook | Features | use_physics_loss / physics_weight | Purpose |
|---|---|---|---|
| `Binary_Classification_Baseline.ipynb` | `PhysicsFeatureExtractor`'s ~36 engineered features | False / 0.0 | ordinary binary classifier, no physics loss |
| `Binary_Classification_PhysicsWeight.ipynb` (formerly misnamed `..._PINN_Algo.ipynb`) | same engineered features | True / 0.1 | this project's own physics-consistency loss term (Mechanism A — not from the reference paper) |
| `Binary_Classification_PINN_Algo.ipynb` (built) | `PaperFeatureSelector`'s tau-threshold-selected frequency-bin features | False / 0.0 (matches paper's own methodology — the paper's "physics-informed" part is the feature *selection*, Algorithm 1, not a loss term) | reference-paper-style feature engineering (Mechanism B) |

All three must share: same `stratified_group_split(..., seed=17)`, same
`compute_class_weights()` formula, same `max_windows_per_file`, same
`hidden_dim`/`epochs`/`batch_size`, so that any metric differences are
attributable to the one intended variable (loss formulation or feature set),
not incidental hyperparameter drift.

## Bugs found in the two existing binary notebooks (fix before running)

**`Binary_Classification_Baseline.ipynb`**: `plt` (matplotlib) is never
imported anywhere, but the final SHAP cell calls `plt.tight_layout()` /
`plt.savefig()` / `plt.show()`. Add `import matplotlib.pyplot as plt` early
(e.g. in the first cell).

**`Binary_Classification_PINN_Algo.ipynb`** (the original one, testing the
physics-weight loss) — this file has since been **renamed to
`Binary_Classification_PhysicsWeight.ipynb`** (it tests this project's own
physics-weight loss term, Mechanism A, not the paper's Algorithm 1 — see
model.py's disambiguation note). The two bugs previously logged here
(missing training cell before the SHAP cell; a leftover
`paper_feature_names` reference) were found to already be fixed in the
notebook content: the full training + `write_evaluation_artifacts()`
pipeline is present in one cell, `model`/`test_X`/`test_y`/`results_dir`
are all defined there before the SHAP cell runs, and the
`plot_loss_vs_feature` cell already uses `feature_names`. No further action
needed on this notebook beyond the `use_physics_loss` rename and updating
its self-description (both done).

The filename `Binary_Classification_PINN_Algo.ipynb` is now used by a
**new** notebook that actually exercises the reference paper's Algorithm 1
(`PaperFeatureSelector` feature selection) — see the comparison table
above and "Paper-algorithm binary notebook (built)" below.

## Paper-algorithm binary notebook (built)

`Binary_Classification_PINN_Algo.ipynb` now implements this: same pipeline
as `Binary_Classification_Baseline.ipynb`, but with `PaperFeatureSelector`
in place of `PhysicsFeatureExtractor`, fit on the training split only
(`PaperFeatureSelector(tau=1.0).fit(windows[train_idx], labels[train_idx])`,
then `.transform(windows)` on the full array), `use_physics_loss=False`,
`physics_weight=0.0`, and outputs saved to `paderborn_binary_paper.npz` /
`results_binary_paper`. `binary_physics_targets` is still passed through
(required by the model's loss function signature even at weight 0.0),
computed via a throwaway `PhysicsFeatureExtractor` call whose 36 engineered
features are discarded — only its `physics_targets` output is used.

## User has just said "I have sufficient time now"

Earlier work today was done under a 2-hour deadline, so several shortcuts
were taken deliberately and documented as limitations rather than fixed:

1. **`fast_fit()` bypasses the 25-combination grid search** — fixed,
   guessed hyperparameters (`learning_rate=0.015`, `physics_weight=0.20`)
   were used instead of tuned ones. **Now that time isn't tight, prefer
   `model.fit()` (the real grid search) for the actual final comparison
   runs**, or at minimum run a cheap grid search on a data subsample (see
   below) to get real tuned values instead of guesses.
2. **`max_windows_per_file=30`** was used instead of fuller coverage.
   Earlier same-day experiments (4-class, before the binary pivot) showed
   accuracy and feature-importance rankings were stable between 16 and 60
   windows/file — so 30–60 is defensible, but exhaustive coverage (~123
   windows/file for a ~4s recording at this window size/stride) was never
   tried and is now affordable time-wise if desired.
3. **`PaperFeatureSelector.top_k=20`** is a fixed guess substituting for the
   paper's threshold-τ selection criterion — worth trying an actual τ sweep
   now, or at least trying a couple different top_k values to check
   sensitivity.
4. **SHAP was run on small samples** (~100–150 test windows,
   `KernelExplainer`/`shap.Explainer` with a small background set) purely
   for speed. Can be scaled up now.

A subsample-based grid search approach was proposed (run the real 25-combo
search on ~3000 training windows instead of the full set, extract
`model.learning_rate` / `model.physics_weight` after fitting, then reuse
those values for full-data training via `fast_fit`) — this was not yet
executed. Consider whether to do this, or just run the full grid search
directly on the full data now that time allows.

## Immediate next steps (suggested)

1. ~~Fix the two known bugs in the existing binary notebooks~~ — the
   Baseline notebook's missing `matplotlib` import is still open; the
   `PhysicsWeight` notebook's two bugs turned out to be already fixed (see
   above).
2. Decide with the user: full grid search per model, or subsample-search-
   then-reuse, given more time is available now.
3. ~~Build the missing paper-algorithm binary notebook~~ — built, see
   "Paper-algorithm binary notebook (built)" above.
4. Run all three binary notebooks to completion, save all three
   `metrics.json` / `confusion_matrix.csv` / `feature_importance.csv`
   outputs.
5. Build a final comparison: a metrics heatmap (accuracy, precision,
   recall, F1 for the binary "faulty" class) across the three notebooks,
   plus a SHAP-vs-weight-magnitude importance comparison for at least the
   Baseline and PINN notebooks (same model, two importance methods).
6. Optionally address the bearing-geometry (6205 vs 6203) caveat if time
   allows — would strengthen the "physics-informed" claims in the report.
7. Any local code changes need to be re-zipped and re-uploaded to Kaggle as
   a new dataset (the "New Version" button on the existing
   `pinn-motor-fault-code` / `-v2` dataset could not be located in the
   Kaggle UI during this session — uploading as a further new dataset,
   e.g. `-v3`, was the working fallback) before a Kaggle notebook will see
   the changes. Update the `cp -r` cell's source path accordingly in
   whichever notebook is used next.

## Known workflow gotchas (avoid repeating)

- Kaggle's `/kaggle/working/` is **wiped when a session ends/restarts**
  (this happened at least twice today, once losing ~1 hour of work). Click
  **Save Version → Quick Save** to persist outputs once a run completes;
  don't rely on `/kaggle/working/` surviving indefinitely.
- After editing a `.py` file that's already been `import`ed in a live
  Kaggle kernel, the change won't take effect until the kernel is
  restarted (Python caches imported modules) — a `pip install -e` alone
  is not enough.
- `PhysicsFeatureExtractor` / `load_paderborn_windows` re-processing the
  full dataset from raw `.mat` files (file I/O + FFT/envelope feature
  extraction) takes several minutes and dominates most cell runtimes —
  don't mistake this for a hang; there's no progress bar on the feature-
  extraction step unless the `tqdm` patch from earlier today is present.
- One `.mat` file is corrupted and always skipped:
  `KA08/N15_M01_F10_KA08_2.mat` ("TypeError: Expecting matrix here") — this
  is expected, not a new bug.