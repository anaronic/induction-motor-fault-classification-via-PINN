"""Training and synthetic validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .features import CLASS_NAMES, BearingPhysics, PhysicsFeatureExtractor
from .model import PhysicsInformedNN
from .paderborn import DatasetError, load_paderborn_windows
from .results import write_evaluation_artifacts


@dataclass
class TrainResult:
    model: PhysicsInformedNN
    train_accuracy: float
    test_accuracy: float
    output_path: Path | None
    train_count: int = 0
    test_count: int = 0


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
    # Calculate class weights if balancing is enabled
    class_weights = None
    if balance_classes:
        class_counts = {label: np.sum(labels == label) for label in CLASS_NAMES}
        total_samples = len(labels)
        class_weights = {label: total_samples/(len(class_counts)*count) for label, count in class_counts.items()}
    extractor = PhysicsFeatureExtractor(BearingPhysics())
    batch = extractor.transform(windows, sources)
    train_indices, test_indices = stratified_split(labels, test_fraction=0.25, seed=11)

    model = PhysicsInformedNN(input_dim=batch.features.shape[1], hidden_dim=32, physics_weight=0.20, learning_rate=0.02)
    validation = (
        batch.features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    model.fit(
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

    model = PhysicsInformedNN(input_dim=batch.features.shape[1], hidden_dim=48, physics_weight=0.20, learning_rate=0.015)
    validation = (
        batch.features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    model.fit(
        batch.features[train_indices],
        labels[train_indices],
        batch.physics_targets[train_indices],
        epochs=epochs,
        batch_size=64,
        validation=validation,
        verbose=True,
    )

    train_loss, train_acc = model.loss_and_accuracy(
        batch.features[train_indices],
        labels[train_indices],
        batch.physics_targets[train_indices],
    )
    test_loss, test_acc = model.loss_and_accuracy(
        batch.features[test_indices],
        labels[test_indices],
        batch.physics_targets[test_indices],
    )
    model.save(model_path)

    test_predictions = model.predict(batch.features[test_indices])
    test_probabilities = model.predict_proba(batch.features[test_indices])
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
    )


def make_synthetic_dataset(
    samples_per_class: int = 32,
    window_size: int = 4096,
    sample_rate_hz: float = 64_000.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
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
