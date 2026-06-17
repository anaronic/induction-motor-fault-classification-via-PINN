# Physics-informed motor fault classification

This project implements a lightweight Physics-Informed Neural Network (PINN/PiNN) pipeline for induction-motor bearing fault classification on the Paderborn University Bearing Dataset.

The implementation follows the project abstract and attached references by combining:

- real Paderborn motor-current/vibration windows,
- signal-processing features from time, spectrum, and envelope spectra,
- bearing characteristic-frequency priors for inner-race, outer-race, and rolling-element defects,
- a trainable neural classifier regularized to stay consistent with physics-derived diagnostic scores.

## Install

```powershell
python -m pip install -e .
```

Install MATLAB `.mat` support when training on extracted Paderborn files:

```powershell
python -m pip install -e .[paderborn]
```

## Pull Paderborn data

The official dataset archives are hosted by the Paderborn KAt Bearing DataCenter:
`https://groups.uni-paderborn.de/kat/BearingDataCenter/`

Download all official bearing-state archives:

```powershell
pinn-motor-fault download --codes K001 K002 K003 K004 K005 K006 KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21 --raw-dir data\paderborn\raw
```

Add `--extract` to unpack archives. Extraction requires a local RAR-capable tool such as `tar`, `7z`, `7za`, `unrar`, or `rar` on `PATH`.

```powershell
pinn-motor-fault download --codes K001 K002 K003 K004 K005 K006 KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21 --extract --raw-dir data\paderborn\raw --extract-dir data\paderborn\extracted
```

## Train

Train and evaluate the final grouped experiment on extracted Paderborn `.mat` files:

```powershell
pinn-motor-fault experiment --data-dir data\paderborn\extracted --epochs 200 --max-windows-per-file 32 --signal vibration --model models\paderborn_final_grouped_pinn.npz --output-dir reports\results_final
```

For a quick end-to-end validation without the full dataset, run:

```powershell
pinn-motor-fault smoke --epochs 10
```

## Show results and visualizations

The `experiment` command above already generates final held-out-file metrics, CSVs, and SVG figures in `reports\results_final`. To evaluate a trained model on all loaded windows separately, run:

```powershell
pinn-motor-fault results --data-dir data\paderborn\extracted --model models\paderborn_final_grouped_pinn.npz --output-dir reports\results_final_all_windows --signal vibration --max-windows-per-file 32
```

Important outputs:

| Output | Purpose |
| --- | --- |
| `reports\results_final\metrics.json` | Held-out-file accuracy and experiment settings |
| `reports\results_final\class_metrics.csv` | Precision, recall, and F1 score per class |
| `reports\results_final\confusion_matrix.csv` | Raw confusion-matrix values |
| `reports\results_final\split_manifest.json` | Exact train/test MATLAB file split |
| `reports\results_final\figures\confusion_matrix.svg` | Confusion-matrix visualization |
| `reports\results_final\figures\signal_examples.svg` | Example vibration windows |
| `reports\results_final\figures\feature_importance.svg` | Most influential physics-informed features |

## Labels

Paderborn bearing codes are mapped as follows:

| Code prefix | Class |
| --- | --- |
| `K00*` | healthy |
| `KI*` | inner race fault |
| `KA*` | outer race fault |
| `KB*` | rolling element fault |
