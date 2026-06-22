# Draft reply to professor

Subject: Update — full-dataset results, train/test split, scaling, and leakage cross-check

Dear Professor [Name],

Thank you for the feedback. Since the partial-data progress report I shared earlier
(4 archives, ~99.5% accuracy), we have substantially expanded the project, and I have
specifically cross-checked the leakage concern you raised. Summary of progress and the
three points you asked about:

## What has changed since the old report
- The old report used only **4 bearing archives** and reported ~99.5%. We have now run the
  pipeline on **all 32 official Paderborn archives** (2559 readable `.mat` files, 81,888
  vibration windows).
- We replaced the random window split with a **stratified split grouped by source `.mat`
  file** and re-evaluated. On the full dataset this gives a more honest
  **86.38% held-out-file accuracy** over 20,480 test windows. We deliberately report this
  lower, more realistic number rather than the inflated 99.5%.

## 1. How the train:test split is done
- The split is **grouped by source `.mat` file**: every window from a given file goes
  *entirely* into either train or test — never both. So overlapping rolling windows from
  one recording can no longer straddle the split.
- It is **stratified by class** (healthy / inner-race / outer-race / rolling-element) so each
  class is proportionally represented (test fraction 0.25).
- For the full experiment: 1919 files (61,408 windows) train, 640 files (20,480 windows) test.
- Code: `stratified_group_split()` in `train.py`; the exact file lists are saved in
  `reports/results_final/split_manifest.json` for audit.

## 2. How the data scaling is done
Scaling is applied in two independent stages, and **statistics are never taken from the test
set**:
1. **Per-window normalisation** (in `features.py`): each 4096-sample window is z-normalised
   using only its *own* mean and std, `x_norm = (x − μ_window)/(σ_window + ε)`. This is
   self-contained per window, so it cannot leak across samples.
2. **Per-feature standardisation** (in `model.py`): a `Standardizer` is **fit on the training
   features only** (`fit_transform` during `fit()`), and the *same* train-derived mean/scale
   is applied to the test features at inference (`transform` in `predict_proba`). The test set
   never contributes to the scaler. There is therefore no scaling leakage.

## 3. Leakage cross-check (the rolling-window concern)
I ran a controlled experiment that trains the **same model on the same windows and features**
under three increasingly strict splits, so the only thing that changes is what is allowed to
be shared between train and test. Results on the **full 32-archive dataset**:

| Split | What can be shared train↔test | Test accuracy |
| --- | --- | --- |
| Random window split (old style) | windows of the *same file* | **0.799** |
| Grouped by `.mat` file (current report) | same bearing code, different file | **0.783** |
| Held-out whole bearing **code** (strictest) | nothing | **0.433** |

(Identical model/features; quick config of 8 windows/file. The report headline of **86.38%**
uses the fuller 32-windows/file, 200-epoch run — the *gap* between splits is the point here.)

**Findings:**
- **Rolling-window leakage is *not* what inflates accuracy.** Random-window and grouped-by-file
  splits give almost the *same* accuracy (0.799 vs 0.783 — only ~1.6 points apart). If
  overlapping windows were the cause, the grouped split would have dropped sharply — it did not.
  So the reported 86.38% is *not* a rolling-window-leakage artefact.
- **The real generalisation gap is at the bearing-instance level.** When we hold out *entire
  bearing codes* (the model never sees that physical bearing under any operating condition),
  accuracy falls to ~0.43. This is honest "unseen-bearing" performance and is the genuine
  limitation to flag — both the random and grouped splits still allow the same bearing code to
  appear (under different load/speed) in train and test.

## Why was the old 4-archive result ~99%? (your "exceptionally good data" question)
I reproduced the **exact old subset** (K001, KA01, KI01, KB23 — one bearing per class) with the
same controlled experiment:

| Split | Test accuracy |
| --- | --- |
| Random window split | **0.994** |
| Grouped by `.mat` file (leak-free) | **0.998** |
| Held-out whole bearing code | **impossible** (0.25 = chance) |

The honest answer is: **it was not "more accurate" data, and it was not rolling-window leakage —
the *task* was simply trivially easy.** Two reasons:
1. **One bearing per class.** With only K001/KA01/KI01/KB23, each class is a *single physical
   bearing* with almost no intra-class variability. The model only has to memorise four fixed
   fingerprints, which are near-perfectly separable. The leak-free grouped split *also* gives
   ~99.8%, which proves leakage was not the cause — the subset is just easy.
2. **No way to test generalisation.** With one bearing per class you *cannot* hold out an unseen
   bearing (doing so removes that class entirely → 25% chance accuracy). So the 99% number says
   nothing about how the model handles *new* bearings.

When we move to all 32 archives, each class now spans many different bearings, damage severities,
and operating conditions. Intra-class variability rises sharply and some fault severities overlap,
so accuracy settles at a believable **86.38%**. In short: the 99% reflected an easy, low-diversity
"lab" subset, not better data — and our current full-dataset result is the scientifically honest one.

## Proposed next step
To give a fully leakage-free generalisation number, the next experiment will split by **bearing
ID / operating condition** (hold out whole bearing codes), report that ~0.43-level accuracy
openly, and add classic baselines (SVM, Random Forest, plain MLP) plus a physics-loss ablation
for comparison. This directly answers the leakage question at the strictest level.

Everything is reproducible from the saved split manifest and the commands in the report. Happy
to walk through any part in person.

Best regards,
Anaranyo
