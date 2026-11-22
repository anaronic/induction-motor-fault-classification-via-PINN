# Physics-Guided Machine Learning for Bearing Fault Diagnosis

## 1. Project Overview

Rolling-element bearing faults can produce characteristic vibration and motor-current signatures before they become visible through routine inspection. This B.Tech research project studies how bearing-domain knowledge can be incorporated into a compact neural-network classification pipeline for the Paderborn University Bearing Dataset.

The project is **not a classical Physics-Informed Neural Network (PINN)**: it does not solve governing differential equations or enforce those equations as constraints. Instead, it investigates four distinct approaches:

1. Generic statistical and spectral features.
2. Diagnostic features located around bearing characteristic frequencies.
3. A signal-derived physics-consistency regularizer during classifier training.
4. A separate Li et al. (2025)-inspired, physics-constrained frequency-bin selection experiment.

The final notebook programme is designed to compare binary and four-class diagnosis, test current/vibration/torque modalities, and separate the effects of diagnostic features from the effects of the training regularizer. The notebooks define experiments; no final experiment results are claimed in this README.

## 2. Research Questions

- How well does a compact neural classifier distinguish healthy from faulty bearings using generic time- and frequency-domain features?
- Does a Li et al.-inspired selection of bearing-frequency bins provide useful binary-classification features?
- How do the current, vibration, and torque modality runs compare when their recording populations and train/test splits are taken into account?
- For four-class diagnosis, what is the effect of adding diagnostic bearing-frequency features?
- With classifier features held fixed, what is the effect of adding the project’s signal-derived physics-consistency loss?
- Do the combined diagnostic-feature and physics-loss settings behave differently from either component in isolation?

## 3. Dataset

The implementation targets the [Paderborn University Bearing DataCenter](https://groups.uni-paderborn.de/kat/BearingDataCenter/) dataset. Its bearing condition is inferred from the MAT-file path or filename using these code prefixes:

| Code prefix | Project label |
|---|---|
| `K00*` | `healthy` |
| `KI*` | `inner_race` |
| `KA*` | `outer_race` |
| `KB*` | `rolling_element` |

The loader supports three named signal preferences. The following rates are the constants configured in this repository; the loader does **not** read or verify a sampling rate from the MAT-file metadata.

| Modality | Configured sampling rate | Duration of 4096 samples | Native FFT-bin spacing |
|---|---:|---:|---:|
| Vibration | 64,000 Hz | 0.064 s | 15.625 Hz |
| Current | 64,000 Hz | 0.064 s | 15.625 Hz |
| Torque | 4,000 Hz | 1.024 s | 0.9766 Hz |

### Data location and loading behaviour

`load_paderborn_windows_with_metadata()` recursively searches the supplied directory for `.mat` files, reads them with `scipy.io.loadmat`, and searches nested MATLAB structures for numeric series. The optional `require_modality_match=True` setting used by the final notebooks requires a channel name containing one of the implementation’s modality tokens; files that cannot be read, labelled, or matched are skipped and reported. Without that option, the loader ranks numeric channels by name and size and may choose a fallback channel.

Each selected signal is divided into overlapping windows. The final notebooks use 4096 samples, stride 2048, and at most 16 windows per recording. They use the beginning of each recording rather than sampling windows across the entire file. No resampling is performed.

The notebook setup cell selects `DATA_DIR` in this order:

1. The `PINN_PADERBORN_DATA` environment variable, if set.
2. `/kaggle/input/paderborn`, if that directory exists.
3. `<project-root>/data/paderborn/extracted`.

The Kaggle default therefore works only when the attached dataset is mounted exactly at `/kaggle/input/paderborn`; otherwise set `PINN_PADERBORN_DATA` to the actual mount. No dataset files are included in this repository.

## 4. Methodology

The final experiment workflow is:

1. **Load recordings.** Read MAT files, infer four-class labels from bearing codes, select the requested signal channel, and retain the source recording for every window.
2. **Window signals.** Use 4096-sample windows with a 2048-sample stride and cap each source recording at 16 windows for the final notebooks.
3. **Create recording-level partitions.** Notebook 00 builds a stratified, held-out-recording vibration split and saves `reports/shared/split_manifest.json`. M1 and M3–M6 reuse this manifest. M2 creates a distinct recording split for each modality.
4. **Extract features or select frequency bins.** Generic and diagnostic features are computed from each window, or the Li-inspired selector is fit using training windows and labels only.
5. **Scale classifier features.** The model’s `Standardizer` is fit on training features and applied to the other partition.
6. **Train a classifier.** A one-hidden-layer NumPy neural network produces class probabilities with a softmax output. The matched M3–M6 settings use 48 hidden units, learning rate 0.015, batch size 64, and 60 epochs.
7. **Evaluate held-out recordings.** Write metrics, per-class results, confusion matrices, predictions, and plots to the experiment’s output directory.

Binary experiments map `healthy` to `healthy` and the three fault labels to `faulty`. Four-class experiments retain `healthy`, `inner_race`, `outer_race`, and `rolling_element`.

## 5. Physics-Guided Components

The code uses a 6203 bearing geometry with \(n=8\) rolling elements, rolling-element diameter \(d=6.75\) mm, pitch diameter \(D=28.55\) mm, and contact angle \(\theta=0^\circ\). For shaft frequency \(f_r\), the implementation derives:

\[
\begin{aligned}
f_{\mathrm{BPFO}} &= \frac{n}{2} f_r
 \left(1-\frac{d}{D}\cos\theta\right),\\
f_{\mathrm{BPFI}} &= \frac{n}{2} f_r
 \left(1+\frac{d}{D}\cos\theta\right),\\
f_{\mathrm{BSF}} &= \frac{D}{2d} f_r
 \left[1-\left(\frac{d}{D}\cos\theta\right)^2\right].
\end{aligned}
\]

At 25 Hz shaft frequency, the configured ratios give BPFO ≈ 76.36 Hz, BPFI ≈ 123.64 Hz, and BSF ≈ 49.92 Hz. Shaft frequency is parsed heuristically from `Nxx` in the source filename (`N15` maps to \(1500/60=25\) Hz); if parsing fails, the code defaults to 25 Hz.

### 5.1 Generic signal features

The generic feature group contains 19 features:

- **12 time-domain descriptors:** mean, standard deviation, RMS, peak amplitude, peak-to-peak value, skewness, kurtosis, crest factor, shape factor, impulse factor, zero-crossing count, and amplitude-distribution entropy.
- **7 spectral descriptors:** spectral centroid, bandwidth, normalized spectral entropy, and four normalized frequency-band energy fractions.

Signals are mean-centered before feature computation. An optional mode also standardizes each window, but the final notebooks use mean-centering alone. Spectrum features use a Hann-windowed one-sided FFT.

### 5.2 Bearing-physics-based diagnostic features

The diagnostic group has 17 features: five envelope-spectrum energy ratios for each of BPFO, BPFI, and BSF harmonics (15 values), plus shaft frequency and the maximum summed fault-harmonic energy ratio. The `all` group concatenates the 19 generic and 17 diagnostic features, giving 36 total.

The harmonic energy ratios use the 5–2000 Hz envelope-spectrum region. A diagnostic harmonic uses a default tolerance of one native FFT bin, \(f_s/N\). For the 4096-sample, 64 kHz window this is 15.625 Hz. This is native FFT resolution; zero-padding would only make a denser displayed frequency grid and would not improve the resolution set by the observation duration.

### 5.3 Physics-consistency regularization

For each raw vibration window, `physics_target_from_vibration_window()` creates a four-component soft target \(q\) independently of the classifier’s feature group and without consulting the ground-truth label:

- Compute the mean-centered window’s analytic envelope and its Hann-windowed spectrum.
- Consider spectrum power from 5 to 2000 Hz.
- Assign each frequency bin within one native FFT-bin tolerance to its nearest BPFO, BPFI, or BSF harmonic, up to five harmonics.
- Assign the remaining power in this range to the healthy component.
- Normalize the four class energies to sum to one.

The target components are ordered as `[healthy, inner_race, outer_race, rolling_element]`. For the binary task they become `[q_healthy, q_inner + q_outer + q_rolling]`. These are **spectral-consistency targets**, not verified health labels or calibrated posterior probabilities.

Conceptually, the classifier objective is:

\[
\mathcal{L} =
\mathcal{L}_{\mathrm{CE}}(y,p)
+ \lambda \frac{1}{C}\sum_{c=1}^{C}(p_c-q_c)^2,
\]

where \(p\) is the softmax output, \(q\) is the signal-derived target, \(C\) is the number of classes, and \(\lambda=0.20\) for M4 and M6. M1, M3, and M5 use cross-entropy only. The training update applies class weights to per-example classification and regularization gradients; the model’s displayed loss calculation is unweighted.

This loss is a project-specific regularizer. It is not part of the Li et al. feature-selection algorithm and does not impose a governing bearing-dynamics equation.

## 6. Li et al. (2025) Reference Method

Li et al.’s Algorithm 1 selects frequency-domain features from nominal and faulty motor spectra. It defines a physics-informed candidate set \(B\), compares z-scored class spectra, computes a difference score \(d[i]=|z_F[i]-z_N[i]|\), and selects candidate bins satisfying \(d[i]\geq\tau\).

The project’s M2 experiment is an **inspired adaptation**, not a reproduction of the paper’s full method:

- It converts the Paderborn labels to binary `healthy`/`faulty`.
- It averages magnitude spectra by class on training windows only, z-scores each class spectrum, computes a class-versus-rest difference, and thresholds at \(\tau=1.0\).
- It forms the candidate bins around bearing-specific BPFO/BPFI/BSF harmonics (five harmonics) and shaft-frequency sidebands, with a 1.5-FFT-bin radius.
- It trains a classifier with cross-entropy only; it does not enable the separate physics-consistency loss.
- The final experiment applies this procedure separately to current, vibration, and torque windows.

The reference paper’s candidate frequencies are based on motor electrical and sideband frequencies; the project substitutes bearing characteristic frequencies and shaft sidebands. Its spectral preprocessing and candidate-set construction therefore differ from the original method. M2 results should be described as Li-inspired rather than as an exact implementation of Algorithm 1.

## 7. Experimental Design

| ID | Notebook | Task and input | Training objective | Data and principal settings |
|---|---|---|---|---|
| M1 | `01_M1_Binary_Baseline.ipynb` | Binary; 19 generic vibration features | Cross-entropy only | 4096 samples, stride 2048, up to 16 windows/recording; seed 17; 48 hidden units; learning rate 0.015; batch 64; 60 epochs |
| M2.1 | `02_M2_Li_Paper_Modality_Comparison.ipynb` | Binary; Li-inspired selected current-frequency bins | Cross-entropy only | 4096 samples; 64 kHz; \(\tau=1.0\); seed 17; 48 hidden units; learning rate 0.015; batch 64; 60 epochs |
| M2.2 | `02_M2_Li_Paper_Modality_Comparison.ipynb` | Binary; Li-inspired selected vibration-frequency bins | Cross-entropy only | Same selector/model settings as M2.1; 64 kHz |
| M2.3 | `02_M2_Li_Paper_Modality_Comparison.ipynb` | Binary; Li-inspired selected torque-frequency bins | Cross-entropy only | Same selector/model settings as M2.1; 4 kHz |
| M3 | `03_M3_Four_Class_Baseline.ipynb` | Four classes; 19 generic vibration features | Cross-entropy only | Shared vibration split; seed 17; 48 hidden units; learning rate 0.015; batch 64; 60 epochs |
| M4 | `04_M4_Physics_Loss_Only.ipynb` | Four classes; 19 generic vibration features | Cross-entropy + physics-consistency loss, \(\lambda=0.20\) | Same split and model/training settings as M3 |
| M5 | `05_M5_Physics_Features_Only.ipynb` | Four classes; 36 generic-plus-diagnostic vibration features | Cross-entropy only | Same split and model/training settings as M3 |
| M6 | `06_M6_Combined_Physics_Model.ipynb` | Four classes; 36 generic-plus-diagnostic vibration features | Cross-entropy + physics-consistency loss, \(\lambda=0.20\) | Same split and model/training settings as M3 |

All notebook experiments use balanced class weights \(w_c=N/(C\,N_c)\), computed from training labels. M1 and M3–M6 write to `reports/M1_binary_baseline`, `reports/M3_four_generic_ce`, `reports/M4_four_generic_physics`, `reports/M5_four_all_ce`, and `reports/M6_four_all_physics`, respectively. M2 writes to `reports/M2_1_current`, `reports/M2_2_vibration`, and `reports/M2_3_torque`.

M2 records a separate source population and split for each modality. Its code flags cross-modality metrics as directly comparable only when both the recording populations and train/test assignments match. Equal sample counts or window sizes alone do not establish comparability.

## 8. Repository Structure

The checked-in project files and current experiment artefacts are:

```text
.
├── 00_Project_Setup_and_Data_Audit.ipynb
├── 01_M1_Binary_Baseline.ipynb
├── 02_M2_Li_Paper_Modality_Comparison.ipynb
├── 03_M3_Four_Class_Baseline.ipynb
├── 04_M4_Physics_Loss_Only.ipynb
├── 05_M5_Physics_Features_Only.ipynb
├── 06_M6_Combined_Physics_Model.ipynb
├── Binary_Classification_Baseline.ipynb
├── Binary_Classification_PhysicsWeight.ipynb
├── Binary_Classification_PINN_Algo.ipynb
├── config/
│   └── experiment_config.json
├── models/                         # currently no committed checkpoints
├── reports/
│   └── results/
│       └── figures/                # existing output directory; no final-run results
├── scripts/
│   ├── generate_notebooks.py
│   └── leakage_check.py
├── src/
│   └── pinn_motor_fault/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── fast_train.py
│       ├── features.py
│       ├── model.py
│       ├── paderborn.py
│       ├── paper_features.py
│       ├── results.py
│       └── train.py
├── tests/
│   ├── test_refactor.py
│   └── test_smoke.py
├── PINN algo ref.pdf
└── pyproject.toml
```

The data directory is excluded from version control and is not present in the current checkout. The three `Binary_Classification_*.ipynb` files are earlier experiments: the baseline and physics-weight notebooks compare a prior engineered-feature setup, while `Binary_Classification_PINN_Algo.ipynb` explores paper-style frequency selection. Their settings and outputs are not interchangeable with the final M1–M6 matrix.

## 9. Installation and Setup

The package requires Python 3.10 or later. `pyproject.toml` declares NumPy (`numpy>=2.0`), a `paderborn` extra for SciPy (`scipy>=1.14`), and a `dev` extra for pytest (`pytest>=8`). The source also imports `tqdm`, but `tqdm` is not declared in the project metadata; install it explicitly until that packaging gap is resolved.

### Windows PowerShell

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[paderborn,dev]" tqdm
python -m pytest -q
```

The package registers the `pinn-motor-fault` command through its `pyproject.toml` entry point. Jupyter itself is not a declared dependency; install and configure Jupyter or the VS Code notebook support separately if needed.

### Kaggle

The final notebooks add `<project-root>/src` to `sys.path`; they do not install the package or download data automatically. The source checkout must be available from the notebook working directory, its parent, or `/kaggle/working/pinn-motor-fault-code-v2`. The data must be attached at `/kaggle/input/paderborn` or configured through `PINN_PADERBORN_DATA`. Ensure the Kaggle environment also has SciPy and `tqdm` available.

## 10. Running Experiments

1. Put extracted Paderborn MAT files in `data\paderborn\extracted` or configure `PINN_PADERBORN_DATA` to the actual directory.
2. Open and run `00_Project_Setup_and_Data_Audit.ipynb` from its first cell. It checks for real MAT recordings, reports the configured frequency resolution, and creates `reports/shared/split_manifest.json`.
3. Run M1 and M3–M6 in fresh kernels using the same project checkout and dataset path. These notebooks require the saved shared manifest and reject mismatched recording populations.
4. Run `02_M2_Li_Paper_Modality_Comparison.ipynb` to execute its current, vibration, and torque runs. Each modality is loaded and split independently; channels without a matching name are skipped.

For a local Windows session, set a custom dataset path **before starting the notebook kernel**:

```powershell
$env:PINN_PADERBORN_DATA = "D:\datasets\paderborn\extracted"
```

Notebook experiments write to the fixed `reports/` directories listed above. Their training helper refuses to overwrite a non-empty experiment directory. Preserve or move prior outputs before rerunning an experiment.

### Optional dataset acquisition through the CLI

The implemented `download` command can retrieve and extract selected archives. This example requests one archive for each label category; it is a subset, not the full dataset:

```powershell
pinn-motor-fault download --codes K001 KA01 KI01 KB23 --extract --raw-dir data\paderborn\raw --extract-dir data\paderborn\extracted
```

Extraction requires an available archive tool recognized by `paderborn.py` (`tar`, `bsdtar`, `7z`, `7za`, `unrar`, or `rar`) and capable of reading the archives. The CLI also offers `pinn-motor-fault smoke --epochs 10` for an explicitly synthetic training smoke test; synthetic examples are not used by the final experiment notebooks.

## 11. Evaluation Metrics and Results

The held-out predictions are evaluated using:

- **Accuracy:** fraction of all predictions that match the reference labels.
- **Macro-F1:** unweighted mean of the per-class F1 scores, giving each class equal contribution.
- **Balanced accuracy:** unweighted mean of per-class recall.
- **Class-wise precision, recall, and F1:** computed from each class’s true positives, false positives, and false negatives.
- **Confusion matrix:** counts actual versus predicted classes, with the CSV’s rows representing actual classes and columns representing predictions.

Each final experiment directory contains `model.npz`, `config.json`, `split_manifest.json`, `metrics.json`, `confusion_matrix.csv`, `class_metrics.csv`, `predictions.csv`, `feature_importance.csv`, and `metrics_table.tex`, plus SVG figures under `figures/`. The M2 directories additionally contain `selected_bins.json` with selected frequencies, difference scores, and feature names. Notebook 00 writes the shared split and `frequency_resolution.json` under `reports/shared/`.

The three older binary notebooks also include SHAP-related cells and may write extra plots/CSVs; SHAP and plotting libraries are not declared project dependencies. The final M1–M6 notebooks do not report SHAP results.

No Paderborn MAT files are present in this checkout, and the final notebooks have not been run on real data here. Consequently, no accuracy or other experiment result is reported.

## 12. Reproducibility and Leakage Prevention

- M1 and M3–M6 use the split saved by Notebook 00: a stratified group split by MAT-file source, test fraction 0.25, seed 17. Manifest use is checked against the loaded recording population, and overlap is rejected.
- M2 creates a recording-level split independently for each modality with seed 17. Because channel availability and successful file parsing can differ, the populations and splits may differ.
- A seed of 17 controls model initialization and epoch minibatch shuffling. The split uses the same seed. M2 fits its feature selector only on training windows and labels.
- The neural model’s feature scaler is fitted using training features and then applied to held-out features.
- Class weights use \(N/(C\,N_c)\) from the training partition.
- M3–M6 match architecture, optimizer step size, batch size, epoch count, class weighting, and split so that their design contrasts feature-group and loss choices.

There is a single fixed seed and one held-out split; the notebooks do not estimate variability across repeated runs. In `fast_fit`, held-out data is passed as validation data and loss/accuracy are printed during training, although those values do not control gradient updates or early stopping. This repeated inspection can influence later human decisions and should be considered when interpreting test metrics.

## 13. Limitations

- **Sampling rates are assumptions.** The loader associates rates with modality names; it does not verify each MAT recording’s rate or resample its signal.
- **Channel identification is heuristic.** Final notebooks require name tokens, but the actual channel naming and availability in the complete dataset have not been validated in this checkout. `M2` torque availability and population remain to be confirmed on the real files.
- **Window coverage is truncated.** The final notebooks use at most the first 16 windows per recording; they do not use every available window or sample windows throughout the recording.
- **Shaft speed is inferred from filenames.** Unrecognized names default to 25 Hz, which can misplace the characteristic-frequency bands.
- **The physics target is a proxy.** It treats spectral energy near ideal characteristic-frequency harmonics as evidence for a class, and all other energy in the selected band as “healthy” energy. It does not account for bearing slip, load-dependent speed changes, transfer paths, or calibrated uncertainty.
- **The loss implementation is lightweight.** The model is a NumPy, one-hidden-layer classifier trained with direct gradient updates. It is not a classical PINN. The conceptual training objective and the unweighted loss values printed by the code do not apply class weights identically.
- **M2 is not a verbatim paper reproduction.** Its candidate set and frequency relationships are adapted to bearing characteristic frequencies, and it has not been validated against the original paper’s experimental protocol.
- **Modality comparisons require matched populations.** Different successful channel matches can yield different recordings; the notebook flags rather than hides this limitation. The window duration also differs substantially for torque due to its configured 4 kHz rate.
- **Some packaging and notebook dependencies are undeclared.** `tqdm` is imported by the package but absent from `pyproject.toml`; Jupyter and the legacy SHAP/plotting dependencies are also not declared there.
- **Final runs are unverified on real data.** The current repository has no dataset or final experiment outputs. Notebook syntax/import smoke checks do not establish full-data correctness or performance.
- **Historical notebooks differ.** Their settings and analysis are not a controlled extension of the current experiment matrix; results from them should not be pooled with M1–M6.

## 14. References

1. S. Li, R. T. Rajan, E. Marth, P. Zorn, W. Gruber, and J. Dauwels, “Physics-informed Intelligent Motor Fault Detection,” in *Proceedings of the 33rd European Signal Processing Conference (EUSIPCO 2025)*, 2025, pp. 1158–1162. The supplied copy is [`PINN algo ref.pdf`](./PINN%20algo%20ref.pdf). M2 is an adaptation of its Algorithm 1, not a verbatim implementation.
2. Paderborn University, KAt BearingDataCenter, “Bearing Data Set.” [https://groups.uni-paderborn.de/kat/BearingDataCenter/](https://groups.uni-paderborn.de/kat/BearingDataCenter/). Dataset archive and bearing-geometry provenance referenced by the loader and feature code.
