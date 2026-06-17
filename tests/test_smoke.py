from pathlib import Path

from pinn_motor_fault.train import make_synthetic_dataset, train_from_windows


def test_synthetic_pipeline_trains(tmp_path: Path) -> None:
    windows, labels, sources = make_synthetic_dataset(samples_per_class=16, window_size=4096)
    result = train_from_windows(windows, labels, sources, output_path=tmp_path / "model.npz", epochs=10)
    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.test_accuracy >= 0.5
