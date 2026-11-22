"""Generate the seven final Paderborn experiment notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(lines: list[str]) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def write_notebook(name: str, purpose: str, cells: list[dict[str, object]]) -> None:
    notebook = {
        "cells": [
            markdown(f"# {name}\n\n{purpose}"),
            code([
                "from pathlib import Path",
                "import json",
                "import os",
                "import sys",
                "",
                "ROOT_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/pinn-motor-fault-code-v2')]",
                "PROJECT_ROOT = next((path.resolve() for path in ROOT_CANDIDATES if (path / 'src' / 'pinn_motor_fault').is_dir()), None)",
                "if PROJECT_ROOT is None:",
                "    raise FileNotFoundError('Cannot locate the repository source tree. Set the notebook working directory to the project root.')",
                "sys.path.insert(0, str(PROJECT_ROOT / 'src'))",
                "REPORTS_DIR = PROJECT_ROOT / 'reports'",
                "if os.environ.get('PINN_PADERBORN_DATA'):",
                "    DATA_DIR = Path(os.environ['PINN_PADERBORN_DATA']).expanduser().resolve()",
                "else:",
                "    KAGGLE_DATA_DIR = Path('/kaggle/input/paderborn')",
                "    LOCAL_DATA_DIR = PROJECT_ROOT / 'data' / 'paderborn' / 'extracted'",
                "    DATA_DIR = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.is_dir() else LOCAL_DATA_DIR",
                "print('Project root:', PROJECT_ROOT)",
                "print('Paderborn data:', DATA_DIR)",
            ]),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")


audit_cells = [
    code([
        "from pinn_motor_fault.features import BearingPhysics, PhysicsFeatureExtractor",
        "from pinn_motor_fault.paderborn import MODALITY_SAMPLE_RATES_HZ, load_paderborn_windows_with_metadata",
        "from pinn_motor_fault.train import assert_group_split_no_overlap, stratified_group_split",
        "",
        "WINDOW_SIZE = 4096",
        "STRIDE = 2048",
        "MAX_WINDOWS_PER_RECORDING = 16",
        "SEED = 17",
        "physics = BearingPhysics(sample_rate_hz=MODALITY_SAMPLE_RATES_HZ['vibration'])",
        "print('6203 characteristic-frequency ratios:', {",
        "    'BPFO': physics.bpfo_ratio, 'BPFI': physics.bpfi_ratio,",
        "    'BSF': physics.bsf_ratio, 'FTF': physics.ftf_ratio,",
        "})",
        "print('At 25 Hz shaft speed:', physics.characteristic_frequencies(25.0))",
        "print('Generic vibration FFT resolution:', physics.frequency_resolution_hz(WINDOW_SIZE), 'Hz')",
    ]),
    markdown(
        "## Real Paderborn audit and shared vibration split\n"
        "This notebook loads actual MAT recordings and saves the exact vibration recording split used by M1 and M3-M6. "
        "Loading failures are allowed to propagate; no synthetic replacement is used."
    ),
    code([
        "assert DATA_DIR.is_dir(), f'Paderborn data directory does not exist: {DATA_DIR}'",
        "loaded = load_paderborn_windows_with_metadata(",
        "    DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE,",
        "    signal_preference='vibration', max_windows_per_file=MAX_WINDOWS_PER_RECORDING,",
        "    require_modality_match=True,",
        ")",
        "assert loaded.modality == 'vibration'",
        "assert loaded.windows.size > 0 and len(loaded.sources) == loaded.labels.size",
        "assert all(Path(source).suffix.lower() == '.mat' for source in loaded.sources), 'Expected real Paderborn MAT recordings.'",
        "data_root = DATA_DIR.resolve()",
        "assert all(Path(source).resolve().is_relative_to(data_root) for source in loaded.sources), 'Loaded recordings must be under DATA_DIR.'",
        "train_indices, test_indices = stratified_group_split(",
        "    loaded.labels, loaded.sources, test_fraction=0.25, seed=SEED,",
        ")",
        "assert_group_split_no_overlap(loaded.sources, train_indices, test_indices)",
        "recordings = sorted(set(str(Path(source)) for source in loaded.sources))",
        "manifest = {",
        "    'dataset': 'real_paderborn', 'modality': loaded.modality,",
        "    'window_size': WINDOW_SIZE, 'stride': STRIDE,",
        "    'sampling_rate_hz': MODALITY_SAMPLE_RATES_HZ[loaded.modality],",
        "    'seed': SEED, 'test_fraction': 0.25,",
        "    'strategy': 'stratified_group_by_mat_file',",
        "    'source_recordings': recordings,",
        "    'train_files': sorted(set(loaded.sources[i] for i in train_indices)),",
        "    'test_files': sorted(set(loaded.sources[i] for i in test_indices)),",
        "    'window_count': int(loaded.labels.size),",
        "    'skipped_recordings': loaded.skipped,",
        "}",
        "shared_dir = REPORTS_DIR / 'shared'",
        "shared_dir.mkdir(parents=True, exist_ok=True)",
        "manifest_path = shared_dir / 'split_manifest.json'",
        "if manifest_path.exists():",
        "    existing = json.loads(manifest_path.read_text(encoding='utf-8'))",
        "    if existing != manifest:",
        "        raise FileExistsError(f'Existing shared split differs; refusing to replace it: {manifest_path}')",
        "else:",
        "    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')",
        "print(json.dumps({",
        "    'modality': loaded.modality, 'windows': int(loaded.labels.size),",
        "    'recordings': len(recordings), 'train_recordings': len(manifest['train_files']),",
        "    'test_recordings': len(manifest['test_files']), 'skipped': len(loaded.skipped),",
        "    'manifest': str(manifest_path),",
        "}, indent=2))",
    ]),
    markdown(
        "## Frequency-resolution report\n"
        "The 4096-sample window remains the generic baseline. The longer sizes below are diagnostic-only resolution "
        "scenarios, not a change to the generic experiment. Zero-padding changes displayed bin spacing but does not "
        "increase true resolution, which is determined by the unpadded observation duration."
    ),
    code([
        "window_sizes = (4096, 8192, 16384, 32768)",
        "resolution_report = {",
        "    modality: [",
        "        {",
        "            'window_size_samples': size,",
        "            'duration_seconds': size / rate,",
        "            'true_fft_resolution_hz': rate / size,",
        "            'purpose': 'generic_baseline' if size == 4096 else 'diagnostic_investigation_only',",
        "        }",
        "        for size in window_sizes",
        "    ]",
        "    for modality, rate in MODALITY_SAMPLE_RATES_HZ.items()",
        "}",
        "resolution_path = shared_dir / 'frequency_resolution.json'",
        "resolution_path.write_text(json.dumps(resolution_report, indent=2), encoding='utf-8')",
        "print(json.dumps(resolution_report, indent=2))",
    ]),
    markdown("## Explicit synthetic smoke test (not an experiment)\nSet the flag only to check feature extraction in a test environment without Paderborn data. This section never feeds synthetic samples to an experiment or writes experiment results."),
    code([
        "RUN_SYNTHETIC_SMOKE_TEST = False",
        "if RUN_SYNTHETIC_SMOKE_TEST:",
        "    from pinn_motor_fault.train import make_synthetic_dataset",
        "    smoke_windows, smoke_labels, smoke_sources = make_synthetic_dataset(samples_per_class=2, window_size=WINDOW_SIZE)",
        "    smoke_batch = PhysicsFeatureExtractor(feature_group='generic').transform(smoke_windows, smoke_sources)",
        "    assert smoke_batch.features.shape == (8, 19)",
        "    assert smoke_batch.physics_targets.shape == (8, 4)",
        "    print('Synthetic feature-extraction smoke test passed; no experiment was run.')",
    ]),
]
write_notebook(
    "00_Project_Setup_and_Data_Audit.ipynb",
    "Audit real Paderborn data, report corrected 6203 frequency resolution, and freeze the shared recording-level split.",
    audit_cells,
)


experiment_specs = [
    {
        "file": "01_M1_Binary_Baseline.ipynb",
        "id": "M1_binary_baseline",
        "purpose": "M1: binary cross-entropy baseline using generic 19-feature vibration input.",
        "task": "binary",
        "feature_group": "generic",
        "physics_loss": False,
    },
    {
        "file": "03_M3_Four_Class_Baseline.ipynb",
        "id": "M3_four_generic_ce",
        "purpose": "M3: four-class generic 19-feature vibration baseline with cross-entropy only.",
        "task": "four",
        "feature_group": "generic",
        "physics_loss": False,
    },
    {
        "file": "04_M4_Physics_Loss_Only.ipynb",
        "id": "M4_four_generic_physics",
        "purpose": "M4: generic 19-feature vibration input with the shared signal-derived physics-consistency loss.",
        "task": "four",
        "feature_group": "generic",
        "physics_loss": True,
    },
    {
        "file": "05_M5_Physics_Features_Only.ipynb",
        "id": "M5_four_all_ce",
        "purpose": "M5: four-class 36-feature vibration input with cross-entropy only.",
        "task": "four",
        "feature_group": "all",
        "physics_loss": False,
    },
    {
        "file": "06_M6_Combined_Physics_Model.ipynb",
        "id": "M6_four_all_physics",
        "purpose": "M6: 36-feature vibration input with the same signal-derived physics-consistency loss as M4.",
        "task": "four",
        "feature_group": "all",
        "physics_loss": True,
    },
]

physics_target_markdown = markdown(
    "## Shared physics-target definition\n"
    "For every raw vibration window, independently of classifier feature group, the target partitions 5-2000 Hz "
    "envelope-spectrum power. Healthy is the fraction outside the BPFO/BPFI/BSF harmonic bands. Inner-race, "
    "outer-race, and rolling-element components are the fractions assigned to the nearest corresponding harmonic "
    "within one native FFT-bin tolerance. The four nonnegative energies are normalized to sum to one. This is a "
    "label-independent spectral-consistency target, not a ground-truth label or calibrated posterior. M4 and M6 "
    "call the same target function on the same raw vibration windows. Binary experiments map it to "
    "`[p_healthy, p_inner + p_outer + p_rolling]`."
)

for spec in experiment_specs:
    cells: list[dict[str, object]] = [
        markdown(
            "## Reproducible configuration\n"
            "The full experiment runs only on real Paderborn recordings. The exact recording population and shared "
            "split are required; missing inputs and loader errors are not caught or replaced with synthetic data."
        ),
    ]
    if spec["physics_loss"]:
        cells.append(physics_target_markdown)
    cells.append(code([
        "from pinn_motor_fault.paderborn import MODALITY_SAMPLE_RATES_HZ, load_paderborn_windows_with_metadata",
        "from pinn_motor_fault.train import run_experiment",
        "",
        f"EXPERIMENT_ID = {spec['id']!r}",
        f"TASK = {spec['task']!r}",
        f"FEATURE_GROUP = {spec['feature_group']!r}",
        f"USE_PHYSICS_LOSS = {spec['physics_loss']!r}",
        "WINDOW_SIZE = 4096",
        "STRIDE = 2048",
        "MAX_WINDOWS_PER_RECORDING = 16",
        "SEED = 17",
        "HIDDEN_DIM = 48",
        "LEARNING_RATE = 0.015",
        "BATCH_SIZE = 64",
        "EPOCHS = 60",
        "assert DATA_DIR.is_dir(), f'Paderborn data directory does not exist: {DATA_DIR}'",
        "loaded = load_paderborn_windows_with_metadata(",
        "    DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE,",
        "    signal_preference='vibration', max_windows_per_file=MAX_WINDOWS_PER_RECORDING,",
        "    require_modality_match=True,",
        ")",
        "assert loaded.modality == 'vibration'",
        "assert loaded.windows.size > 0 and len(loaded.sources) == loaded.labels.size",
        "assert all(Path(source).suffix.lower() == '.mat' for source in loaded.sources), 'Expected real Paderborn MAT recordings.'",
        "data_root = DATA_DIR.resolve()",
        "assert all(Path(source).resolve().is_relative_to(data_root) for source in loaded.sources), 'Loaded recordings must be under DATA_DIR.'",
        "split_manifest = REPORTS_DIR / 'shared' / 'split_manifest.json'",
        "assert split_manifest.is_file(), 'Run 00_Project_Setup_and_Data_Audit.ipynb first to create the shared real-recording split.'",
        "experiment_dir = REPORTS_DIR / EXPERIMENT_ID",
        "metrics = run_experiment(",
        "    loaded.windows, loaded.labels, loaded.sources, experiment_dir,",
        "    task=TASK, feature_group=FEATURE_GROUP,",
        "    use_physics_loss=USE_PHYSICS_LOSS, physics_weight=0.20 if USE_PHYSICS_LOSS else 0.0,",
        "    hidden_dim=HIDDEN_DIM, learning_rate=LEARNING_RATE,",
        "    batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED,",
        "    split_manifest=split_manifest, experiment_id=EXPERIMENT_ID,",
        "    dataset_name='real_paderborn_vibration',",
        ")",
        "print(json.dumps({key: metrics[key] for key in ('accuracy', 'macro_f1', 'balanced_accuracy')}, indent=2))",
        "print('Artifacts:', experiment_dir.resolve())",
    ]))
    cells.append(markdown(
        "## Output artifacts\n"
        "Each fixed experiment ID writes its own model checkpoint, configuration, exact split, metrics, predictions, "
        "CSV summaries, and figures. The training code refuses to overwrite a non-empty experiment directory."
    ))
    write_notebook(spec["file"], str(spec["purpose"]), cells)


m2_cells = [
    markdown(
        "## Complete three-modality experiment\n"
        "Each modality is loaded, split by recording, selected using training recordings only, trained, evaluated, "
        "and saved to its own immutable experiment directory. Populations and split manifests are recorded per "
        "modality. The results are directly comparable only when both the recording populations and train/test "
        "recording assignments match."
    ),
    code([
        "from pinn_motor_fault.paderborn import MODALITY_SAMPLE_RATES_HZ, load_paderborn_windows_with_metadata",
        "from pinn_motor_fault.train import run_paper_feature_experiment",
        "",
        "WINDOW_SIZE = 4096",
        "STRIDE = 2048",
        "MAX_WINDOWS_PER_RECORDING = 16",
        "SEED = 17",
        "EPOCHS = 60",
        "BATCH_SIZE = 64",
        "MODALITY_EXPERIMENTS = {",
        "    'current': 'M2_1_current',",
        "    'vibration': 'M2_2_vibration',",
        "    'torque': 'M2_3_torque',",
        "}",
        "assert DATA_DIR.is_dir(), f'Paderborn data directory does not exist: {DATA_DIR}'",
        "comparison = {}",
        "for modality, experiment_id in MODALITY_EXPERIMENTS.items():",
        "    loaded = load_paderborn_windows_with_metadata(",
        "        DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE,",
        "        signal_preference=modality, max_windows_per_file=MAX_WINDOWS_PER_RECORDING,",
        "        require_modality_match=True,",
        "    )",
        "    assert loaded.modality == modality",
        "    assert loaded.windows.size > 0 and len(loaded.sources) == loaded.labels.size",
        "    assert all(Path(source).suffix.lower() == '.mat' for source in loaded.sources), 'Expected real Paderborn MAT recordings.'",
        "    data_root = DATA_DIR.resolve()",
        "    assert all(Path(source).resolve().is_relative_to(data_root) for source in loaded.sources), 'Loaded recordings must be under DATA_DIR.'",
        "    experiment_dir = REPORTS_DIR / experiment_id",
        "    metrics = run_paper_feature_experiment(",
        "        loaded.windows, loaded.labels, loaded.sources, experiment_dir,",
        "        modality=modality, sample_rate_hz=MODALITY_SAMPLE_RATES_HZ[modality],",
        "        skipped=loaded.skipped, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED,",
        "        dataset_name='real_paderborn',",
        "    )",
        "    split = json.loads((experiment_dir / 'split_manifest.json').read_text(encoding='utf-8'))",
        "    config = json.loads((experiment_dir / 'config.json').read_text(encoding='utf-8'))",
        "    comparison[modality] = {",
        "        'experiment_id': experiment_id,",
        "        'accuracy': metrics['accuracy'],",
        "        'macro_f1': metrics['macro_f1'],",
        "        'balanced_accuracy': metrics['balanced_accuracy'],",
        "        'source_recordings': config['source_recordings'],",
        "        'train_files': split['train_files'],",
        "        'test_files': split['test_files'],",
        "        'frequency_resolution_hz': config['frequency_resolution_hz'],",
        "    }",
        "reference = comparison['current']",
        "for modality, record in comparison.items():",
        "    same_population = set(record['source_recordings']) == set(reference['source_recordings'])",
        "    same_split = record['train_files'] == reference['train_files'] and record['test_files'] == reference['test_files']",
        "    record['same_population_as_current'] = same_population",
        "    record['same_split_as_current'] = same_split",
        "    record['directly_comparable_to_current'] = same_population and same_split",
        "print(json.dumps(comparison, indent=2))",
        "print('Modality runs have separate source populations and split manifests; interpret cross-modality metrics only where directly_comparable_to_current is true.')",
    ]),
]
write_notebook(
    "02_M2_Li_Paper_Modality_Comparison.ipynb",
    "M2: complete Li-inspired binary feature-selection experiments for current, vibration, and torque.",
    m2_cells,
)
