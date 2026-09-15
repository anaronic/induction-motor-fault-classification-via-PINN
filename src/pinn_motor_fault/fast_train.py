"""Lightweight training entry point that skips PhysicsInformedNN.fit()'s
25-combination hyperparameter grid search (5 learning rates x 5 physics
weights, each run for the full epoch count) for time-constrained runs.
"""

from __future__ import annotations

import numpy as np

from .model import PhysicsInformedNN, TrainingHistory, labels_to_indices


def compute_class_weights(labels: np.ndarray, class_names: tuple[str, ...]) -> dict[str, float]:
    """Same inverse-frequency + focal-loss formula used elsewhere in this
    project (train.py's train_from_windows/run_grouped_experiment), factored
    out here so every trained model in the 4-version comparison uses an
    identical weighting scheme -- otherwise accuracy/F1 differences could be
    partly explained by inconsistent class balancing rather than the
    feature set or importance method under test.
    """
    class_counts = {label: int(np.sum(labels == label)) for label in class_names}
    total = len(labels)
    return {
        label: (total / (len(class_counts) * count)) * (1 - (count / total)) ** 2
        for label, count in class_counts.items()
    }


def fast_fit(
    model: PhysicsInformedNN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    physics_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    physics_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    class_weights: dict[str, float] | None = None,
    verbose: bool = True,
    print_every: int = 5,
) -> TrainingHistory:
    """Train without the grid search. Mirrors PhysicsInformedNN.fit()'s
    core epoch loop (standardize, iterate, evaluate) minus the
    hyperparameter search, so the resulting model is directly comparable
    to one trained via .fit() with the same explicit hyperparameters.
    """
    x_train_scaled = model.standardizer.fit_transform(x_train)
    x_val_scaled = model.standardizer.transform(x_val)
    y_train_idx = labels_to_indices(y_train, model.class_names)
    y_val_idx = labels_to_indices(y_val, model.class_names)

    history = TrainingHistory(loss=[], accuracy=[])
    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc = model._train_epoch(
            x_train_scaled, y_train_idx, physics_train, batch_size, class_weights=class_weights
        )
        history.loss.append(epoch_loss)
        history.accuracy.append(epoch_acc)
        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            val_loss, val_acc = model._evaluate_indices(x_val_scaled, y_val_idx, physics_val)
            print(
                f"epoch={epoch:03d} loss={epoch_loss:.4f} accuracy={epoch_acc:.3f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_acc:.3f}"
            )

    model.training_history = history
    return history