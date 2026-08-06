"""Evaluation metrics and lightweight SVG visualizations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .features import CLASS_NAMES, PhysicsFeatureExtractor
from .model import PhysicsInformedNN, TrainingHistory
from .paderborn import load_paderborn_windows


def generate_results(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    window_size: int = 4096,
    stride: int = 2048,
    signal_preference: str = "current",
    max_windows_per_file: int | None = 8,
) -> dict[str, object]:
    # Load data, extract features, run the saved model, and write evaluation artifacts.
    windows, labels, sources = load_paderborn_windows(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        signal_preference=signal_preference,
        max_windows_per_file=max_windows_per_file,
    )
    feature_batch = PhysicsFeatureExtractor().transform(windows, sources)
    model = PhysicsInformedNN.load(model_path)
    predictions = model.predict(feature_batch.features)
    probabilities = model.predict_proba(feature_batch.features)
    settings = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "sample_count": int(labels.size),
        "window_size": int(window_size),
        "stride": int(stride),
        "signal": signal_preference,
        "split_strategy": "all_loaded_windows",
    }
    return write_evaluation_artifacts(
        output_dir=output_dir,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        sources=sources,
        model=model,
        feature_names=feature_batch.feature_names,
        windows=windows,
        signal_name=signal_preference,
        settings=settings,
        training_history=model.training_history,
    )


def write_evaluation_artifacts(
    output_dir: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    sources: list[str],
    model: PhysicsInformedNN,
    feature_names: list[str],
    windows: np.ndarray,
    signal_name: str,
    settings: dict[str, object],
    include_hyperparams: bool = True,
    training_history: TrainingHistory | None = None,
) -> dict[str, object]:
    # Generate JSON, CSV, and SVG evaluation outputs for a trained model.
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    confusion = confusion_matrix(labels, predictions, model.class_names)
    class_metrics = per_class_metrics(confusion, model.class_names)
    accuracy = float(np.mean(predictions == labels))

    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "class_names": list(model.class_names),
        "class_distribution": {
            name: int(np.sum(labels == name)) 
            for name in model.class_names
        },
    }
    if include_hyperparams:
        metrics.update({
            "hyperparameters": {
                "learning_rate": float(model.learning_rate),
                "physics_weight": float(model.physics_weight),
                "hidden_dim": int(model.hidden_dim),
                "batch_size": settings.get("batch_size", 64),
            }
        })
    metrics.update(settings)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_csv(output_dir / "confusion_matrix.csv", confusion, model.class_names)
    write_class_metrics_csv(output_dir / "class_metrics.csv", class_metrics)
    write_prediction_csv(output_dir / "predictions.csv", labels, predictions, probabilities, sources, model.class_names)
    write_feature_importance(output_dir / "feature_importance.csv", figure_dir / "feature_importance.svg", model, feature_names)
    write_confusion_svg(figure_dir / "confusion_matrix.svg", confusion, model.class_names)
    write_signal_examples_svg(figure_dir / "signal_examples.svg", windows, labels, signal_name)
    if training_history is not None:
        write_training_curves_svg(figure_dir / "training_curves.svg", training_history)
        metrics["training_history"] = {
            "epochs": len(training_history.loss),
            "loss": training_history.loss,
            "accuracy": training_history.accuracy,
        }
    write_latex_metrics(output_dir / "metrics_table.tex", metrics)
    return metrics


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, class_names: tuple[str, ...]) -> np.ndarray:
    # Build a confusion matrix from true and predicted label pairs.
    index = {label: i for i, label in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[index[str(actual)], index[str(predicted)]] += 1
    return matrix


def per_class_metrics(confusion: np.ndarray, class_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    # Compute precision, recall, and F1 score for each class.
    metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = float(confusion[i, i])
        fp = float(np.sum(confusion[:, i]) - tp)
        fn = float(np.sum(confusion[i, :]) - tp)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def write_confusion_csv(path: Path, confusion: np.ndarray, class_names: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, confusion):
            writer.writerow([name, *row.tolist()])


def write_class_metrics_csv(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1"])
        for name, values in metrics.items():
            writer.writerow([name, f"{values['precision']:.4f}", f"{values['recall']:.4f}", f"{values['f1']:.4f}"])


def write_prediction_csv(
    path: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    sources: list[str],
    class_names: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "actual", "predicted", *[f"p_{name}" for name in class_names]])
        for source, actual, predicted, probability in zip(sources, labels, predictions, probabilities):
            writer.writerow([source, actual, predicted, *[f"{value:.6f}" for value in probability]])


def write_feature_importance(csv_path: Path, svg_path: Path, model: PhysicsInformedNN, feature_names: list[str]) -> None:
    # Estimate feature importance from model weights and save ranking with a bar plot.
    class_weight = np.mean(np.abs(model.w2), axis=1)
    scores = np.abs(model.w1) @ class_weight
    order = np.argsort(scores)[::-1]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "importance"])
        for rank, index in enumerate(order, start=1):
            writer.writerow([rank, feature_names[index], f"{scores[index]:.6f}"])
    top = order[:12]
    labels = [feature_names[index] for index in top]
    values = scores[top]
    write_bar_svg(svg_path, labels, values, "Top physics-informed feature importances")


def write_confusion_svg(path: Path, confusion: np.ndarray, class_names: tuple[str, ...]) -> None:
    size = 86
    left = 150
    top = 70
    width = left + size * len(class_names) + 40
    height = top + size * len(class_names) + 100
    max_value = max(1, int(confusion.max()))
    cells = []
    for i, actual in enumerate(class_names):
        y = top + i * size
        cells.append(f'<text x="{left - 12}" y="{y + 48}" text-anchor="end" font-size="13">{_esc(actual)}</text>')
        for j, predicted in enumerate(class_names):
            x = left + j * size
            value = int(confusion[i, j])
            shade = 245 - int(170 * value / max_value)
            cells.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="rgb({shade},{shade},255)" stroke="#333"/>')
            cells.append(f'<text x="{x + size / 2}" y="{y + 50}" text-anchor="middle" font-size="18">{value}</text>')
    headers = [
        f'<text x="{left + j * size + size / 2}" y="{top - 15}" text-anchor="middle" font-size="13">{_esc(name)}</text>'
        for j, name in enumerate(class_names)
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">Confusion matrix</text>
<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-size="14">Predicted class</text>
<text x="20" y="{top + size * len(class_names) / 2}" transform="rotate(-90 20 {top + size * len(class_names) / 2})" text-anchor="middle" font-size="14">Actual class</text>
{''.join(headers)}
{''.join(cells)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_signal_examples_svg(path: Path, windows: np.ndarray, labels: np.ndarray, signal_name: str) -> None:
    selected: list[tuple[str, np.ndarray]] = []
    for class_name in CLASS_NAMES:
        indices = np.flatnonzero(labels == class_name)
        if indices.size:
            selected.append((class_name, windows[int(indices[0])][:512]))
    width = 900
    row_height = 130
    height = 70 + row_height * len(selected)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">Example {signal_name} windows</text>')
    for row, (name, signal) in enumerate(selected):
        y0 = 55 + row * row_height
        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
        normalized = np.clip(normalized, -4, 4)
        points = []
        for i, value in enumerate(normalized):
            x = 140 + i * (width - 180) / max(1, normalized.size - 1)
            y = y0 + row_height / 2 - value * 14
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<text x="20" y="{y0 + row_height / 2:.1f}" font-size="14">{_esc(name)}</text>')
        lines.append(f'<line x1="140" y1="{y0 + row_height / 2:.1f}" x2="{width - 30}" y2="{y0 + row_height / 2:.1f}" stroke="#ddd"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1565c0" stroke-width="1.4"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bar_svg(path: Path, labels: list[str], values: np.ndarray, title: str) -> None:
    width = 900
    row_height = 34
    height = 70 + row_height * len(labels)
    max_value = float(np.max(values) + 1e-12)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">{_esc(title)}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 55 + i * row_height
        bar_width = 600 * float(value) / max_value
        lines.append(f'<text x="20" y="{y + 18}" font-size="12">{_esc(label[:34])}</text>')
        lines.append(f'<rect x="260" y="{y}" width="{bar_width:.1f}" height="22" fill="#2e7d32"/>')
        lines.append(f'<text x="{270 + bar_width:.1f}" y="{y + 17}" font-size="12">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_training_curves_svg(path: Path, history: TrainingHistory) -> None:
    loss_values = np.asarray(history.loss, dtype=np.float64)
    accuracy_values = np.asarray(history.accuracy, dtype=np.float64)
    epochs = np.arange(1, len(loss_values) + 1)
    if len(epochs) == 0:
        return

    width = 900
    height = 420
    left = 70
    top = 60
    right = width - 40
    bottom = height - 70
    plot_width = right - left
    plot_height = bottom - top
    max_loss = float(np.max(loss_values) + 1e-12)
    lines: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">Training curves</text>')
    lines.append(f'<text x="{width / 2}" y="52" text-anchor="middle" font-size="13">Loss and accuracy over epochs</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')

    x_positions = [left + (epoch - 1) / max(1, len(epochs) - 1) * plot_width for epoch in epochs]
    loss_points = [f"{x:.1f},{top + plot_height * (1.0 - float(value) / max_loss):.1f}" for x, value in zip(x_positions, loss_values)]
    acc_points = [f"{x:.1f},{top + plot_height * (1.0 - float(value)):.1f}" for x, value in zip(x_positions, accuracy_values)]
    lines.append(f'<polyline points="{" ".join(loss_points)}" fill="none" stroke="#d32f2f" stroke-width="2"/>')
    lines.append(f'<polyline points="{" ".join(acc_points)}" fill="none" stroke="#1976d2" stroke-width="2"/>')
    lines.append(f'<text x="{left + 8}" y="{top + 16}" font-size="12" fill="#d32f2f">Loss</text>')
    lines.append(f'<text x="{left + 68}" y="{top + 16}" font-size="12" fill="#1976d2">Accuracy</text>')
    lines.append(f'<text x="{left}" y="{bottom + 24}" font-size="12">Epoch 1</text>')
    lines.append(f'<text x="{right - 40}" y="{bottom + 24}" font-size="12">Epoch {len(epochs)}</text>')
    lines.append(f'<text x="{left - 50}" y="{top + 6}" font-size="11">High</text>')
    lines.append(f'<text x="{left - 50}" y="{bottom + 4}" font-size="11">Low</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_metrics(path: Path, metrics: dict[str, object]) -> None:
    class_metrics = metrics["class_metrics"]
    assert isinstance(class_metrics, dict)
    rows = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Class & Precision & Recall & F1 \\",
        r"\hline",
    ]
    for name, values in class_metrics.items():
        escaped_name = name.replace("_", r"\_")
        rows.append(f"{escaped_name} & {values['precision']:.3f} & {values['recall']:.3f} & {values['f1']:.3f} \\\\")
    rows.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
