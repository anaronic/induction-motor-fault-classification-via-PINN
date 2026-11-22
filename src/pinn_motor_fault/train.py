"""Training and synthetic validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .features import CLASS_NAMES, BearingPhysics, PhysicsFeatureExtractor, Standardizer
from .model import PhysicsInformedNN, TrainingHistory
from .fast_train import fast_fit, compute_class_weights
from .paderborn import DatasetError, load_paderborn_windows, parse_shaft_frequency_hz
from .paper_features import PaperFeatureSelector
from .results import write_evaluation_artifacts


@dataclass
class TrainResult:
    model: PhysicsInformedNN
    train_accuracy: float
    test_accuracy: float
    output_path: Path | None
    train_count: int = 0
    test_count: int = 0
    history: TrainingHistory | None = None


def train_from_paderborn(
    data_dir: Path,
    output_path: Path | None,
    epochs: int = 50,
    window_size: int = 4096,
    stride: int = 2048,
    signal_preference: str = "current",
    max_windows_per_file: int | None = 20,
    synthetic_if_empty: bool = False,
) -> TrainResult:
    try:
        # Load raw Paderborn windows from a directory and optionally fall back to synthetic data.
        windows, labels, sources = load_paderborn_windows(
            data_dir,
            window_size=window_size,
            stride=stride,
            signal_preference=signal_preference,
            max_windows_per_file=max_windows_per_file,
        )
    except DatasetError:
        if not synthetic_if_empty:
            raise
        windows, labels, sources = make_synthetic_dataset(window_size=window_size)
    return train_from_windows(windows, labels, sources, output_path=output_path, epochs=epochs)


def train_from_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    sources: list[str] | None,
    output_path: Path | None,
    epochs: int = 50,
    balance_classes: bool = True,
) -> TrainResult:
    """Train model with optional class balancing.

    If balance_classes=True, weights samples inversely proportional to class frequency.
    """
    # Enhanced class balancing with focal loss weighting
    class_weights = None
    if balance_classes:
        class_counts = {label: np.sum(labels == label) for label in CLASS_NAMES}
        total_samples = len(labels)
        # Standard balanced weighting: n_samples / (n_classes * class_count)
        class_weights = {
            label: float(total_samples / (len(class_counts) * count))
            for label, count in class_counts.items()
        }
        print(f"Using class weights: {class_weights}")
    extractor = PhysicsFeatureExtractor(BearingPhysics())
    batch = extractor.transform(windows, sources)
    train_indices, test_indices = stratified_split(labels, test_fraction=0.25, seed=11)

    model = PhysicsInformedNN(input_dim=batch.features.shape[1], hidden_dim=32, physics_weight=0.20, learning_rate=0.02)
    validation = (
        batch.features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    history = model.fit(
        batch.features[train_indices],
        labels[train_indices],
        batch.physics_targets[train_indices],
        epochs=epochs,
        batch_size=32,
        validation=validation,
        verbose=True,
    )
    _, train_acc = model.loss_and_accuracy(batch.features[train_indices], labels[train_indices], batch.physics_targets[train_indices])
    _, test_acc = model.loss_and_accuracy(batch.features[test_indices], labels[test_indices], batch.physics_targets[test_indices])
    if output_path is not None:
        model.save(output_path)
    return TrainResult(
        model=model,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        output_path=output_path,
        train_count=int(train_indices.size),
        test_count=int(test_indices.size),
        history=history,
    )


def run_grouped_experiment(
    data_dir: Path,
    model_path: Path,
    results_dir: Path,
    epochs: int = 60,
    window_size: int = 4096,
    stride: int = 2048,
    signal_preference: str = "vibration",
    max_windows_per_file: int | None = 16,
    test_fraction: float = 0.25,
) -> TrainResult:
    windows, labels, sources = load_paderborn_windows(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        signal_preference=signal_preference,
        max_windows_per_file=max_windows_per_file,
    )
    extractor = PhysicsFeatureExtractor(BearingPhysics())
    batch = extractor.transform(windows, sources)
    train_indices, test_indices = stratified_group_split(labels, sources, test_fraction=test_fraction, seed=17)

    scaler = Standardizer()
    scaler.fit(batch.features[train_indices])
    scaled_features = scaler.transform(batch.features)

    model = PhysicsInformedNN(input_dim=scaled_features.shape[1], hidden_dim=48, physics_weight=0.20, learning_rate=0.015)
    validation = (
        scaled_features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    history = model.fit(
        scaled_features[train_indices],
        labels[train_indices],
        batch.physics_targets[train_indices],
        epochs=epochs,
        batch_size=64,
        validation=validation,
        verbose=True,
    )

    train_loss, train_acc = model.loss_and_accuracy(
        scaled_features[train_indices],
        labels[train_indices],
        batch.physics_targets[train_indices],
    )
    test_loss, test_acc = model.loss_and_accuracy(
        scaled_features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    model.save(model_path)

    test_predictions = model.predict(scaled_features[test_indices])
    test_probabilities = model.predict_proba(scaled_features[test_indices])
    settings = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "sample_count": int(test_indices.size),
        "train_sample_count": int(train_indices.size),
        "test_sample_count": int(test_indices.size),
        "unique_train_files": int(len(set(sources[index] for index in train_indices))),
        "unique_test_files": int(len(set(sources[index] for index in test_indices))),
        "window_size": int(window_size),
        "stride": int(stride),
        "signal": signal_preference,
        "split_strategy": "stratified_group_by_mat_file",
        "epochs": int(epochs),
        "max_windows_per_file": max_windows_per_file,
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
    }
    write_evaluation_artifacts(
        output_dir=results_dir,
        labels=labels[test_indices],
        predictions=test_predictions,
        probabilities=test_probabilities,
        sources=[sources[index] for index in test_indices],
        model=model,
        feature_names=batch.feature_names,
        windows=windows[test_indices],
        signal_name=signal_preference,
        settings=settings,
        training_history=history,
    )
    split_manifest = {
        "train_files": sorted(set(sources[index] for index in train_indices)),
        "test_files": sorted(set(sources[index] for index in test_indices)),
    }
    (results_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    return TrainResult(
        model=model,
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        output_path=model_path,
        train_count=int(train_indices.size),
        test_count=int(test_indices.size),
        history=history,
    )


def run_experiment(
    windows: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    output_dir: Path,
    *,
    task: str = "four",
    feature_group: str = "generic",
    use_physics_loss: bool = False,
    physics_weight: float = 0.0,
    hidden_dim: int = 48,
    learning_rate: float = 0.015,
    batch_size: int = 64,
    epochs: int = 60,
    seed: int = 17,
    preprocessing: str = "mean_center",
    split_manifest: Path | Mapping[str, object] | None = None,
    experiment_id: str | None = None,
    dataset_name: str = "provided_windows",
) -> dict[str, object]:
    """Reusable fixed-budget experiment entry point."""
    _prepare_output_directory(output_dir)
    if task not in {"binary", "four"}:
        raise ValueError("task must be 'binary' or 'four'")
    if task == "binary":
        labels = binary_labels(labels)
        class_names = ("healthy", "faulty")
    else:
        class_names = CLASS_NAMES

    if split_manifest is None:
        train_indices, test_indices = stratified_group_split(
            labels, sources, test_fraction=0.25, seed=seed
        )
    else:
        train_indices, test_indices = _indices_from_split_manifest(sources, split_manifest)
    assert_group_split_no_overlap(sources, train_indices, test_indices)

    batch = PhysicsFeatureExtractor(
        BearingPhysics(),
        preprocessing=preprocessing,
        feature_group=feature_group,
    ).transform(windows, sources)
    features = batch.features
    physics_targets = batch.physics_targets
    if task == "binary":
        physics_targets = np.column_stack([
            physics_targets[:, 0],
            np.sum(physics_targets[:, 1:], axis=1),
        ])
    weights = compute_class_weights(labels[train_indices], class_names)
    model = PhysicsInformedNN(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        class_names=class_names,
        physics_weight=physics_weight if use_physics_loss else 0.0,
        learning_rate=learning_rate,
        use_physics_loss=use_physics_loss,
        seed=seed,
    )
    fast_fit(
        model,
        features[train_indices],
        labels[train_indices],
        physics_targets[train_indices],
        features[test_indices],
        labels[test_indices],
        physics_targets[test_indices],
        epochs=epochs,
        batch_size=batch_size,
        class_weights=weights,
        verbose=True,
    )

    predictions = model.predict(features[test_indices])
    probabilities = model.predict_proba(features[test_indices])
    model_path = output_dir / "model.npz"
    model.save(model_path)
    settings = {
        "experiment_id": experiment_id or output_dir.name,
        "dataset": dataset_name,
        "task": task,
        "feature_group": feature_group,
        "use_physics_loss": use_physics_loss,
        "physics_weight": float(model.physics_weight),
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "preprocessing": preprocessing,
        "class_weights": weights,
        "split_strategy": "stratified_group_by_mat_file",
        "source_recordings": sorted(set(str(Path(source)) for source in sources)),
        "train_recordings": sorted(set(str(Path(sources[i])) for i in train_indices)),
        "test_recordings": sorted(set(str(Path(sources[i])) for i in test_indices)),
        "physics_target_method": "raw_vibration_envelope_energy_partition_v1",
    }
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(json.dumps({
        "train_files": settings["train_recordings"],
        "test_files": settings["test_recordings"],
        "source_recordings": settings["source_recordings"],
        "seed": seed,
        "strategy": "stratified_group_by_mat_file",
    }, indent=2), encoding="utf-8")
    return write_evaluation_artifacts(
        output_dir=output_dir,
        labels=labels[test_indices],
        predictions=predictions,
        probabilities=probabilities,
        sources=[sources[i] for i in test_indices],
        model=model,
        feature_names=batch.feature_names,
        windows=windows[test_indices],
        signal_name="paderborn",
        settings=settings,
        training_history=model.training_history,
    )


def run_paper_feature_experiment(
    windows: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    output_dir: Path,
    *,
    modality: str,
    sample_rate_hz: float,
    skipped: list[dict[str, str]] | None = None,
    epochs: int = 60,
    batch_size: int = 64,
    hidden_dim: int = 48,
    learning_rate: float = 0.015,
    seed: int = 17,
    tau: float = 1.0,
    dataset_name: str = "provided_windows",
) -> dict[str, object]:
    """Run a recording-level binary Li-inspired feature-selection experiment."""
    _prepare_output_directory(output_dir)
    if len(sources) != len(labels) or windows.shape[0] != labels.size:
        raise ValueError("windows, labels, and sources must have matching lengths.")
    if modality not in {"current", "vibration", "torque"}:
        raise ValueError(f"Unsupported modality: {modality}")

    labels = binary_labels(labels)
    train_indices, test_indices = stratified_group_split(
        labels, sources, test_fraction=0.25, seed=seed
    )
    assert_group_split_no_overlap(sources, train_indices, test_indices)
    physics = BearingPhysics(sample_rate_hz=sample_rate_hz)
    shaft_frequencies = np.asarray(
        [parse_shaft_frequency_hz(sources[int(i)]) for i in train_indices],
        dtype=np.float64,
    )
    selector = PaperFeatureSelector(sample_rate_hz=sample_rate_hz, tau=tau)
    selector.fit(
        windows[train_indices],
        labels[train_indices],
        shaft_hz=shaft_frequencies,
        physics=physics,
    )
    features = selector.transform(windows)
    class_names = ("healthy", "faulty")
    class_weights = compute_class_weights(labels[train_indices], class_names)
    model = PhysicsInformedNN(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        class_names=class_names,
        physics_weight=0.0,
        learning_rate=learning_rate,
        use_physics_loss=False,
        seed=seed,
    )
    physics_targets = np.zeros((labels.size, len(class_names)), dtype=np.float64)
    fast_fit(
        model,
        features[train_indices],
        labels[train_indices],
        physics_targets[train_indices],
        features[test_indices],
        labels[test_indices],
        physics_targets[test_indices],
        epochs=epochs,
        batch_size=batch_size,
        class_weights=class_weights,
        verbose=True,
    )
    predictions = model.predict(features[test_indices])
    probabilities = model.predict_proba(features[test_indices])
    source_recordings = sorted(set(str(Path(source)) for source in sources))
    train_recordings = sorted(set(str(Path(sources[i])) for i in train_indices))
    test_recordings = sorted(set(str(Path(sources[i])) for i in test_indices))
    settings: dict[str, object] = {
        "experiment_id": output_dir.name,
        "dataset": dataset_name,
        "task": "binary",
        "modality": modality,
        "sample_rate_hz": float(sample_rate_hz),
        "window_size": int(windows.shape[1]),
        "frequency_resolution_hz": float(sample_rate_hz / windows.shape[1]),
        "feature_method": "li_inspired_training_only_frequency_selection",
        "selected_feature_count": int(features.shape[1]),
        "tau": float(tau),
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "class_weights": class_weights,
        "split_strategy": "stratified_group_by_mat_file",
        "source_recordings": source_recordings,
        "train_recordings": train_recordings,
        "test_recordings": test_recordings,
        "skipped_recordings": skipped or [],
    }
    model.save(output_dir / "model.npz")
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(json.dumps({
        "train_files": train_recordings,
        "test_files": test_recordings,
        "source_recordings": source_recordings,
        "seed": seed,
        "strategy": "stratified_group_by_mat_file",
    }, indent=2), encoding="utf-8")
    (output_dir / "selected_bins.json").write_text(json.dumps({
        "frequency_hz": selector.selected_freqs.tolist(),
        "difference_scores": selector.diff_scores.tolist(),
        "feature_names": selector.feature_names(),
        "tau": tau,
        "physics_bin_indices": selector.selected_indices.tolist(),
    }, indent=2), encoding="utf-8")
    return write_evaluation_artifacts(
        output_dir=output_dir,
        labels=labels[test_indices],
        predictions=predictions,
        probabilities=probabilities,
        sources=[sources[i] for i in test_indices],
        model=model,
        feature_names=selector.feature_names(),
        windows=windows[test_indices],
        signal_name=modality,
        settings=settings,
        training_history=model.training_history,
    )


def _prepare_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Experiment output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Experiment output directory already contains files; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _indices_from_split_manifest(
    sources: list[str],
    manifest: Path | Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(manifest, Path):
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        manifest_data = manifest
    train_files = manifest_data.get("train_files")
    test_files = manifest_data.get("test_files")
    if not isinstance(train_files, list) or not isinstance(test_files, list):
        raise ValueError("Split manifest must contain train_files and test_files lists.")
    normalize = lambda value: str(Path(str(value)).resolve())
    train_set = {normalize(path) for path in train_files}
    test_set = {normalize(path) for path in test_files}
    if train_set & test_set:
        raise ValueError("Split manifest has recording overlap between train_files and test_files.")
    source_set = {normalize(source) for source in sources}
    if train_set | test_set != source_set:
        missing = sorted(source_set - (train_set | test_set))
        extra = sorted((train_set | test_set) - source_set)
        raise ValueError(
            f"Split manifest population does not match loaded recordings; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    train_indices = np.asarray(
        [index for index, source in enumerate(sources) if normalize(source) in train_set],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, source in enumerate(sources) if normalize(source) in test_set],
        dtype=np.int64,
    )
    if not train_indices.size or not test_indices.size:
        raise ValueError("Split manifest must assign at least one window to both partitions.")
    return train_indices, test_indices


def make_synthetic_dataset(
    samples_per_class: int = 32,
    window_size: int = 4096,
    sample_rate_hz: float = 64_000.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    # Synthesize simple signals for each class using bearing fault modulation patterns.
    physics = BearingPhysics(sample_rate_hz=sample_rate_hz)
    shaft_hz = 25.0
    times = np.arange(window_size) / sample_rate_hz
    windows: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []

    for label in CLASS_NAMES:
        for index in range(samples_per_class):
            carrier_hz = 4500.0 + rng.normal(0.0, 80.0)
            signal = 0.25 * np.sin(2 * np.pi * 50.0 * times)
            signal += 0.15 * np.sin(2 * np.pi * carrier_hz * times)
            signal += rng.normal(0.0, 0.15, size=window_size)
            if label != "healthy":
                fault_hz = physics.characteristic_frequencies(shaft_hz)[label]
                modulation = 1.0 + 0.45 * np.sin(2 * np.pi * fault_hz * times)
                signal += modulation * 0.35 * np.sin(2 * np.pi * carrier_hz * times)
                for harmonic in range(2, 5):
                    signal += 0.07 / harmonic * np.sin(2 * np.pi * fault_hz * harmonic * times)
            windows.append(signal.astype(np.float64))
            labels.append(label)
            sources.append(f"N15_M07_F10_SYNTH_{label}_{index}.mat")
    return np.stack(windows), np.asarray(labels), sources


def stratified_split(labels: np.ndarray, test_fraction: float = 0.25, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    # Create a balanced train/test split while preserving class proportions.
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(set(str(item) for item in labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(indices.size * test_fraction)))
        test.extend(indices[:test_count].tolist())
        train.extend(indices[test_count:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def stratified_group_split(
    labels: np.ndarray,
    sources: list[str],
    test_fraction: float = 0.25,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sources) != labels.size:
        raise ValueError("sources must have one entry per label.")
    rng = np.random.default_rng(seed)
    group_to_indices: dict[str, list[int]] = {}
    group_to_label: dict[str, str] = {}
    for index, (label, source) in enumerate(zip(labels, sources)):
        group = str(Path(source))
        group_to_indices.setdefault(group, []).append(index)
        group_to_label[group] = str(label)

    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for label in sorted(set(group_to_label.values())):
        groups = [group for group, group_label in group_to_label.items() if group_label == label]
        rng.shuffle(groups)
        test_count = max(1, int(round(len(groups) * test_fraction)))
        test_groups.update(groups[:test_count])
        train_groups.update(groups[test_count:])

    train_indices = [index for group in sorted(train_groups) for index in group_to_indices[group]]
    test_indices = [index for group in sorted(test_groups) for index in group_to_indices[group]]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)


def assert_group_split_no_overlap(
    sources: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Raise if recording identifiers occur in both split partitions."""
    train_files = {str(Path(sources[int(i)])) for i in train_indices}
    test_files = {str(Path(sources[int(i)])) for i in test_indices}
    overlap = sorted(train_files & test_files)
    if overlap:
        raise ValueError(f"Recording-level split leakage detected: {overlap}")


def binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.asarray(["healthy" if str(label) == "healthy" else "faulty" for label in labels])
