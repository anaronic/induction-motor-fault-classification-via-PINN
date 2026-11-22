import math

import numpy as np
from pathlib import Path

from pinn_motor_fault.features import (
    BearingGeometry6203,
    BearingPhysics,
    PhysicsFeatureExtractor,
    physics_target_from_vibration_window,
)
from pinn_motor_fault.fast_train import compute_class_weights
from pinn_motor_fault.model import PhysicsInformedNN
from pinn_motor_fault.paderborn import MODALITY_SAMPLE_RATES_HZ, parse_shaft_frequency_hz
from pinn_motor_fault.train import (
    assert_group_split_no_overlap,
    binary_labels,
    run_experiment,
    stratified_group_split,
)


def test_6203_geometry_ratios_and_25_hz() -> None:
    physics = BearingPhysics()
    assert math.isclose(physics.bpfo_ratio, 3.054290718, rel_tol=1e-8)
    assert math.isclose(physics.bpfi_ratio, 4.945709282, rel_tol=1e-8)
    assert math.isclose(physics.bsf_ratio, 1.996601155, rel_tol=1e-8)
    assert math.isclose(physics.ftf_ratio, 0.381786339, rel_tol=1e-8)
    frequencies = physics.characteristic_frequencies(25.0)
    assert math.isclose(frequencies["outer_race"], 76.3573, abs_tol=1e-3)
    assert math.isclose(frequencies["inner_race"], 123.6427, abs_tol=1e-3)
    assert math.isclose(frequencies["rolling_element"], 49.9150, abs_tol=1e-3)


def test_feature_group_sizes_and_resolution() -> None:
    windows = np.random.default_rng(7).normal(size=(4, 4096))
    physics = BearingPhysics()
    generic = PhysicsFeatureExtractor(physics, preprocessing="mean_center", feature_group="generic").transform(
        windows, ["N15_M07_F10_test.mat"] * 4
    )
    all_features = PhysicsFeatureExtractor(physics, feature_group="all").transform(
        windows, ["N15_M07_F10_test.mat"] * 4
    )
    assert generic.features.shape[1] == 19
    assert all_features.features.shape[1] == 36
    assert generic.physics_targets.shape == (4, 4)
    assert all_features.physics_targets.shape == (4, 4)
    assert physics.frequency_resolution_hz(4096) == 15.625
    assert physics.diagnostic_tolerance_hz(4096) == 15.625


def test_physics_targets_are_signal_derived_and_feature_group_independent() -> None:
    physics = BearingPhysics()
    shaft_hz = 25.0
    window_size = 4096
    times = np.arange(window_size) / physics.sample_rate_hz
    inner_race_hz = physics.characteristic_frequencies(shaft_hz)["inner_race"]
    carrier = 5000.0
    vibration = (
        1.0 + 0.9 * np.sin(2.0 * np.pi * inner_race_hz * times)
    ) * np.sin(2.0 * np.pi * carrier * times)
    sources = ["N15_M07_F10_inner_test.mat"]

    generic = PhysicsFeatureExtractor(
        physics, feature_group="generic"
    ).transform(vibration[None, :], sources)
    all_features = PhysicsFeatureExtractor(
        physics, feature_group="all"
    ).transform(vibration[None, :], sources)
    target = physics_target_from_vibration_window(vibration, shaft_hz, physics)

    assert np.allclose(generic.physics_targets, all_features.physics_targets)
    assert np.allclose(generic.physics_targets[0], target)
    assert np.isclose(target.sum(), 1.0)
    assert target[1] > target[2]
    assert target[1] > target[3]


def test_mean_center_and_standardize_are_distinct() -> None:
    window = np.arange(4096, dtype=np.float64)
    for strategy in ("mean_center", "mean_center_standardize"):
        batch = PhysicsFeatureExtractor(preprocessing=strategy, feature_group="generic").transform(
            window[None, :], ["N15_M07_F10_test.mat"]
        )
        assert np.isfinite(batch.features).all()


def test_modality_metadata_and_shaft_frequency() -> None:
    assert MODALITY_SAMPLE_RATES_HZ == {"vibration": 64000.0, "current": 64000.0, "torque": 4000.0}
    assert parse_shaft_frequency_hz("N15_M07_F10.mat") == 25.0
    assert parse_shaft_frequency_hz("N09_M07_F10.mat") == 15.0


def test_balanced_class_weights_are_standard() -> None:
    weights = compute_class_weights(np.asarray(["healthy"] * 9 + ["faulty"] * 3), ("healthy", "faulty"))
    assert weights["healthy"] == 2.0 / 3.0
    assert weights["faulty"] == 2.0


def test_recording_group_split_has_no_overlap() -> None:
    labels = np.asarray(["healthy"] * 8 + ["faulty"] * 8)
    sources = [f"/data/K00_{i}.mat" if i < 8 else f"/data/KA{i}.mat" for i in range(16)]
    train, test = stratified_group_split(labels, sources, test_fraction=0.25, seed=17)
    assert_group_split_no_overlap(sources, train, test)

    overlap_sources = sources[:2] + sources[:2]
    overlap_labels = np.asarray(["healthy", "healthy", "healthy", "healthy"])
    try:
        assert_group_split_no_overlap(overlap_sources, np.asarray([0, 1]), np.asarray([2, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected overlap sources to be rejected")


def test_binary_labels_and_smoke_models(tmp_path: Path) -> None:
    labels = binary_labels(np.asarray(["healthy", "inner_race", "outer_race", "rolling_element"]))
    assert set(labels.tolist()) == {"healthy", "faulty"}
    binary_model = PhysicsInformedNN(3, class_names=("healthy", "faulty"), use_physics_loss=False)
    x = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    y = np.asarray(["healthy", "faulty"])
    history = binary_model.fit(x, y, np.asarray([[1.0, 0.0], [0.0, 1.0]]), epochs=2)
    assert len(history.loss) == 2

    from pinn_motor_fault.train import make_synthetic_dataset, run_experiment
    windows, labels, sources = make_synthetic_dataset(samples_per_class=4, window_size=4096)
    result = run_experiment(
        windows, labels, sources, tmp_path,
        task="binary", feature_group="generic", epochs=2, batch_size=16,
    )
    assert "accuracy" in result


def test_training_reproducibility_includes_minibatch_order() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 5))
    labels = np.asarray(["healthy", "inner_race", "outer_race", "rolling_element"] * 6)
    targets = np.full((24, 4), 0.25)
    models = [
        PhysicsInformedNN(input_dim=5, hidden_dim=8, seed=17, use_physics_loss=False)
        for _ in range(2)
    ]
    for model in models:
        model.fit(x, labels, targets, epochs=4, batch_size=6, verbose=False)
    assert np.array_equal(models[0].w1, models[1].w1)
    assert np.array_equal(models[0].w2, models[1].w2)


def test_experiment_output_directory_is_not_overwritten(tmp_path: Path) -> None:
    from pinn_motor_fault.train import make_synthetic_dataset

    windows, labels, sources = make_synthetic_dataset(samples_per_class=2, window_size=4096)
    output_dir = tmp_path / "M1_binary_baseline"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    try:
        run_experiment(windows, labels, sources, output_dir, task="binary", epochs=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing experiment artifacts to be protected.")
    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_experiment_reuses_exact_recording_manifest(tmp_path: Path) -> None:
    import json

    from pinn_motor_fault.train import make_synthetic_dataset

    windows, labels, sources = make_synthetic_dataset(samples_per_class=4, window_size=4096)
    train_indices, test_indices = stratified_group_split(labels, sources, seed=17)
    manifest = {
        "train_files": sorted(set(sources[i] for i in train_indices)),
        "test_files": sorted(set(sources[i] for i in test_indices)),
    }
    output_dir = tmp_path / "M3_four_generic_ce"
    run_experiment(
        windows,
        labels,
        sources,
        output_dir,
        task="four",
        feature_group="generic",
        epochs=1,
        batch_size=8,
        split_manifest=manifest,
    )
    saved_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["train_files"] == manifest["train_files"]
    assert saved_manifest["test_files"] == manifest["test_files"]


def test_paper_feature_experiment_writes_isolated_artifacts(tmp_path: Path) -> None:
    from pinn_motor_fault.train import make_synthetic_dataset, run_paper_feature_experiment

    windows, labels, sources = make_synthetic_dataset(
        samples_per_class=8, window_size=4096, seed=3
    )
    output_dir = tmp_path / "M2_1_current"
    metrics = run_paper_feature_experiment(
        windows,
        labels,
        sources,
        output_dir,
        modality="current",
        sample_rate_hz=64_000.0,
        epochs=2,
        batch_size=16,
        seed=17,
    )
    assert {"accuracy", "macro_f1", "balanced_accuracy"} <= set(metrics)
    for artifact in (
        "model.npz",
        "config.json",
        "metrics.json",
        "predictions.csv",
        "selected_bins.json",
        "split_manifest.json",
        "figures",
    ):
        assert (output_dir / artifact).exists()
