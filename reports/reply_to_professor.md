Subject: Update — full-dataset results + leakage cross-check

Dear Professor [Name],

Thank you for the feedback. Quick update: since the partial 4-archive report (~99.5%), we have
re-run the pipeline on all 32 official Paderborn archives (2559 files, ~82k windows), and I
specifically checked the leakage concern.

- **Train/test split:** stratified, **grouped by source `.mat` file** — every window from a file
  goes entirely to train or test, so overlapping windows can't straddle the split. Exact file
  lists are saved in `split_manifest.json`.
- **Scaling:** done in two stages with no test-set leakage — each window is z-normalised by its
  own mean/std, then a feature standardiser is **fit on the training set only** and applied to
  test.
- **Leakage cross-check:** training the same model under different splits, random-window vs
  grouped-by-file differ by only ~1.6 points, so **rolling-window leakage is not inflating the
  result**. On the old 4-archive subset, the leak-free grouped split *also* gives ~99.8% — that
  number was high simply because one bearing per class is a trivially easy task, not because of
  leakage or better data. With all 32 archives the accuracy settles at a realistic **86.38%**.

The genuine limitation we found is bearing-instance generalisation: holding out *whole bearing
codes* drops accuracy to ~0.43. As a next step I'll report that strict split openly and add
SVM/RF/MLP baselines plus a physics-loss ablation. Everything is reproducible; happy to walk
through it.

Best regards,
Anaranyo
