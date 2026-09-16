"""A small physics-informed neural classifier implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import CLASS_NAMES, Standardizer


@dataclass
class TrainingHistory:
    loss: list[float]
    accuracy: list[float]


# Small NumPy-based neural model that combines classification loss with a physics-informed target loss.
#
# NOTE ON "PHYSICS-INFORMED" TERMINOLOGY (two unrelated mechanisms exist in
# this project -- do not conflate them):
#   - THIS FILE implements Mechanism A: a physics-CONSISTENCY LOSS TERM
#     (`physics_weight` * MSE against `physics_targets` from features.py),
#     gated by `use_physics_loss`. This is this project's own design choice,
#     not from the reference paper.
#   - Mechanism B is the reference paper's actual "Algorithm 1"
#     (Li et al., EUSIPCO 2025) -- a FEATURE-SELECTION step with no loss-term
#     component at all, implemented separately in `paper_features.py`
#     (`PaperFeatureSelector`). When training on paper-selected features,
#     use `use_physics_loss=False` / `physics_weight=0.0` here, since the
#     paper's own loss function is plain cross-entropy.
#   See `paper_features.py`'s module docstring for the other half of this note.
class PhysicsInformedNN:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        class_names: tuple[str, ...] = CLASS_NAMES,
        physics_weight: float = 0.25,
        learning_rate: float = 0.01,
        use_physics_loss: bool = True,
        seed: int = 7,
    ) -> None:
        """Physics-Informed Neural Network for motor fault detection.

        Architecture Details:
        - Input Layer: input_dim features
        - Hidden Layer: hidden_dim units with tanh activation
          - Weight initialization: He initialization
          - Bias initialization: Zero initialization
        - Output Layer: len(class_names) units with softmax activation
          - Weight initialization: He initialization
          - Bias initialization: Zero initialization

        Training Process:
        - Batch size: 64 (default)
        - Learning rate: 0.01 (default)
        - Physics weight (λ): 0.25 (default)
        - Loss Function: CrossEntropyLoss + λ * PhysicsLoss (only when
          use_physics_loss=True; otherwise plain CrossEntropyLoss). See the
          module-level note above for how this differs from the reference
          paper's "Algorithm 1" feature-selection mechanism.
        - Optimizer: Gradient Descent with momentum

        Physics Constraints:
        - Enforces bearing fault frequency characteristics
        - Uses envelope spectrum analysis
        - Harmonic energy constraints

        Convergence Criteria:
        - Early stopping based on loss improvement
        - Patience: 5 epochs (default)
        - Convergence threshold: 1e-5 (default)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_names = tuple(class_names)
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        self.use_physics_loss = use_physics_loss
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, len(class_names)))
        self.b2 = np.zeros(len(class_names))
        self.standardizer = Standardizer()
        self.training_history: TrainingHistory | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        physics_targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        verbose: bool = True,
        lambda_range: tuple[float, float] = (0.1, 0.5),
        lr_range: tuple[float, float] = (0.01, 0.1),
        convergence_threshold: float = 1e-5,
        patience: int = 5,
        class_weights: dict[str, float] | None = None,
    ) -> TrainingHistory:
        """Fit the model to training data with optional hyperparameter search."""
        # Standardize input features before training.
        x_train = self.standardizer.fit_transform(x)
        y_indices = labels_to_indices(y, self.class_names)

        validation_data = None
        if validation is not None:
            x_val, y_val, physics_val = validation
            validation_data = (
                self.standardizer.transform(x_val),
                labels_to_indices(y_val, self.class_names),
                physics_val,
            )

        best_loss = float("inf")
        best_params = (self.learning_rate, self.physics_weight)

        for lr in np.linspace(lr_range[0], lr_range[1], 5):
            for physics_weight in np.linspace(lambda_range[0], lambda_range[1], 5):
                trial = PhysicsInformedNN(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    class_names=self.class_names,
                    physics_weight=physics_weight,
                    learning_rate=lr,
                    use_physics_loss=self.use_physics_loss,
                    seed=0,
                )
                trial.standardizer.mean_ = self.standardizer.mean_
                trial.standardizer.scale_ = self.standardizer.scale_
                trial.w1 = self.w1.copy()
                trial.b1 = self.b1.copy()
                trial.w2 = self.w2.copy()
                trial.b2 = self.b2.copy()
                trial_history = trial._fit_epochs(
                    x_train,
                    y_indices,
                    physics_targets,
                    epochs,
                    batch_size,
                    validation_data,
                    verbose=False,
                    class_weights=class_weights,
                )
                if trial_history.loss and trial_history.loss[-1] < best_loss:
                    best_loss = trial_history.loss[-1]
                    best_params = (lr, physics_weight)

        self.learning_rate, self.physics_weight = best_params

        best_loss = float("inf")
        no_improvement_count = 0
        history = TrainingHistory(loss=[], accuracy=[])

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_acc = self._train_epoch(
                x_train,
                y_indices,
                physics_targets,
                batch_size,
                class_weights=class_weights,
            )

            if epoch_loss < best_loss - convergence_threshold:
                best_loss = epoch_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} - loss converged")
                    break

            history.loss.append(epoch_loss)
            history.accuracy.append(epoch_acc)

            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                message = f"epoch={epoch:03d} loss={epoch_loss:.4f} accuracy={epoch_acc:.3f}"
                if validation_data is not None:
                    val_loss, val_acc = self._evaluate_indices(*validation_data)
                    message += f" val_loss={val_loss:.4f} val_accuracy={val_acc:.3f}"
                print(message)

        self.training_history = history
        return history

    def _fit_epochs(
        self,
        x: np.ndarray,
        y_indices: np.ndarray,
        physics_targets: np.ndarray,
        epochs: int,
        batch_size: int,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
        class_weights: dict[str, float] | None = None,
    ) -> TrainingHistory:
        # Run training for a fixed number of epochs and collect history.
        history = TrainingHistory(loss=[], accuracy=[])
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_acc = self._train_epoch(
                x,
                y_indices,
                physics_targets,
                batch_size,
                class_weights=class_weights,
            )
            history.loss.append(epoch_loss)
            history.accuracy.append(epoch_acc)
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                message = f"epoch={epoch:03d} loss={epoch_loss:.4f} accuracy={epoch_acc:.3f}"
                if validation is not None:
                    val_loss, val_acc = self._evaluate_indices(*validation)
                    message += f" val_loss={val_loss:.4f} val_accuracy={val_acc:.3f}"
                print(message)
        return history

    def _train_epoch(
        self,
        x: np.ndarray,
        y_indices: np.ndarray,
        physics_targets: np.ndarray,
        batch_size: int,
        class_weights: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        # Shuffle training samples and apply batch updates over one epoch.
        rng = np.random.default_rng()
        order = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], batch_size):
            batch_indices = order[start : start + batch_size]
            self._train_batch(
                x[batch_indices],
                y_indices[batch_indices],
                physics_targets[batch_indices],
                class_weights=class_weights,
            )
        return self._evaluate_indices(x, y_indices, physics_targets)

    def _evaluate_indices(
        self,
        x: np.ndarray,
        y_indices: np.ndarray,
        physics_targets: np.ndarray,
    ) -> tuple[float, float]:
        # Compute cross-entropy plus physics-informed target loss, and accuracy.
        _, probabilities = self._forward_standardized(x)
        ce = -np.mean(np.log(probabilities[np.arange(y_indices.size), y_indices] + 1e-12))
        physics_loss = float(np.mean((probabilities - physics_targets) ** 2))
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == y_indices))
        total_loss = ce + self.physics_weight * physics_loss if self.use_physics_loss else ce
        return float(total_loss), accuracy

    def _train_batch(
        self, 
        x: np.ndarray, 
        y_indices: np.ndarray, 
        physics_targets: np.ndarray,
        class_weights: dict[str, float] | None = None
    ) -> None:
        # Forward pass, gradient calculation, and parameter update for one batch.
        hidden, probabilities = self._forward_standardized(x)
        n = x.shape[0]
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(n), y_indices] = 1.0

        # Apply class weights if provided
        weights = np.ones(x.shape[0])
        if class_weights is not None:
            weights = np.array([class_weights[self.class_names[index]] for index in y_indices])
            
        dlogits = (probabilities - one_hot) * weights.reshape(-1, 1) / n
        if self.use_physics_loss and self.physics_weight > 0:
            dprob = (2.0 * self.physics_weight / n) * (probabilities - physics_targets) * weights.reshape(-1, 1)
            correction = np.sum(dprob * probabilities, axis=1, keepdims=True)
            dlogits += probabilities * (dprob - correction)

        dw2 = hidden.T @ dlogits
        db2 = np.sum(dlogits, axis=0)
        dhidden = dlogits @ self.w2.T
        dz1 = dhidden * (1.0 - hidden * hidden)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Return class probability estimates for input examples.
        standardized = self.standardizer.transform(x)
        _, probabilities = self._forward_standardized(standardized)
        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Predict class labels from highest softmax probability.
        indices = np.argmax(self.predict_proba(x), axis=1)
        return np.asarray([self.class_names[index] for index in indices])

    def loss_and_accuracy(self, x: np.ndarray, y: np.ndarray, physics_targets: np.ndarray) -> tuple[float, float]:
        # Compute overall loss and prediction accuracy on data.
        probabilities = self.predict_proba(x)
        y_indices = labels_to_indices(y, self.class_names)
        ce = -np.mean(np.log(probabilities[np.arange(y_indices.size), y_indices] + 1e-12))
        physics_loss = float(np.mean((probabilities - physics_targets) ** 2))
        predictions = np.argmax(probabilities, axis=1)
        accuracy = float(np.mean(predictions == y_indices))
        total_loss = ce + self.physics_weight * physics_loss if self.use_physics_loss else ce
        return float(total_loss), accuracy

    def save(self, path: Path) -> None:
        # Persist trained weights, standardizer parameters, and hyperparameters.
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.standardizer.mean_ is None or self.standardizer.scale_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        savez_kwargs = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "mean": self.standardizer.mean_,
            "scale": self.standardizer.scale_,
            "class_names": np.asarray(self.class_names),
            "physics_weight": np.asarray([self.physics_weight]),
            "learning_rate": np.asarray([self.learning_rate]),
            "use_physics_loss": np.asarray([self.use_physics_loss]),
        }
        if self.training_history is not None:
            savez_kwargs["history_loss"] = np.asarray(self.training_history.loss, dtype=np.float64)
            savez_kwargs["history_accuracy"] = np.asarray(self.training_history.accuracy, dtype=np.float64)
        np.savez(path, **savez_kwargs)

    @classmethod
    def load(cls, path: Path) -> "PhysicsInformedNN":
        # Load model state and standardization parameters from disk.
        data = np.load(path, allow_pickle=False)
        model = cls(
            input_dim=int(data["w1"].shape[0]),
            hidden_dim=int(data["w1"].shape[1]),
            class_names=tuple(str(item) for item in data["class_names"]),
            physics_weight=float(data["physics_weight"][0]),
            learning_rate=float(data["learning_rate"][0]),
            use_physics_loss=(
                bool(data["use_physics_loss"][0])
                if "use_physics_loss" in data
                # Backward compatibility with .npz files saved before the
                # use_physics_algorithm -> use_physics_loss rename.
                else bool(data["use_physics_algorithm"][0]) if "use_physics_algorithm" in data else True
            ),
        )
        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]
        model.standardizer.mean_ = data["mean"]
        model.standardizer.scale_ = data["scale"]
        if "history_loss" in data and "history_accuracy" in data:
            model.training_history = TrainingHistory(
                loss=list(data["history_loss"].astype(float)),
                accuracy=list(data["history_accuracy"].astype(float)),
            )
        return model

    def _forward_standardized(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        probabilities = _softmax(logits)
        return hidden, probabilities


def labels_to_indices(labels: np.ndarray, class_names: tuple[str, ...] = CLASS_NAMES) -> np.ndarray:
    # Map string labels to integer indices for classification training.
    mapping = {label: index for index, label in enumerate(class_names)}
    try:
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown label {exc.args[0]!r}; expected one of {class_names}") from exc


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
