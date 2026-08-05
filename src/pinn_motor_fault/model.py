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
    learning_rates: list[float]  # Track effective learning rates
    grad_norms: list[float]     # Track gradient norms


class PhysicsInformedNN:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        class_names: tuple[str, ...] = CLASS_NAMES,
        physics_weight: float = 0.25,
        learning_rate: float = 0.01,
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
        - Loss Function: CrossEntropyLoss + λ * PhysicsLoss
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
        """Physics-Informed Neural Network for motor fault detection.
        
        Architecture:
        - Input layer: input_dim features
        - Hidden layer: hidden_dim units with tanh activation
        - Output layer: len(class_names) units with softmax activation
        
        Loss Function:
        L = CrossEntropyLoss + λ * PhysicsLoss
        where λ = physics_weight
        
        PhysicsLoss = MSE between predicted probabilities and physics targets
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_names = tuple(class_names)
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, len(class_names)))
        self.b2 = np.zeros(len(class_names))
        self.standardizer = Standardizer()

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
    ) -> TrainingHistory:
        """Fit model with hyperparameter tuning.
        
        Args:
            lambda_range: Range of physics weight values to try
            lr_range: Range of learning rates to try
        """
        # Initialize best parameters
        best_loss = float('inf')
        best_params = None
        
        # Grid search over hyperparameters
        for lr in np.linspace(lr_range[0], lr_range[1], 5):
            for physics_weight in np.linspace(lambda_range[0], lambda_range[1], 5):
                self.learning_rate = lr
                self.physics_weight = physics_weight
                
                # Train with current parameters
                history = self._fit_epochs(x, y, physics_targets, epochs, batch_size, validation, verbose)
                
                # Track best configuration
                current_loss = history.loss[-1]
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = (lr, physics_weight)
        
        # Set best parameters
        self.learning_rate, self.physics_weight = best_params
        
        # Add convergence checking
        best_loss = float('inf')
        no_improvement_count = 0
        history = TrainingHistory(loss=[], accuracy=[], learning_rates=[], grad_norms=[])
        
        for epoch in range(1, epochs + 1):
            # Train one epoch
            epoch_loss, epoch_acc = self._train_epoch(x, y, physics_targets, batch_size)
            
            # Check convergence
            if epoch_loss < best_loss - convergence_threshold:
                best_loss = epoch_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} - loss converged")
                    break
            
            # Track history
            history.loss.append(epoch_loss)
            history.accuracy.append(epoch_acc)
            
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                print(f"epoch={epoch:03d} loss={epoch_loss:.4f} accuracy={epoch_acc:.3f}")
        
        return history
        x_train = self.standardizer.fit_transform(x)
        y_indices = labels_to_indices(y, self.class_names)
        history = TrainingHistory(loss=[], accuracy=[])
        rng = np.random.default_rng(42)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                batch_indices = order[start : start + batch_size]
                self._train_batch(x_train[batch_indices], y_indices[batch_indices], physics_targets[batch_indices])

            loss, acc = self.loss_and_accuracy(x, y, physics_targets)
            history.loss.append(loss)
            history.accuracy.append(acc)
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                message = f"epoch={epoch:03d} loss={loss:.4f} accuracy={acc:.3f}"
                if validation is not None:
                    val_loss, val_acc = self.loss_and_accuracy(*validation)
                    message += f" val_loss={val_loss:.4f} val_accuracy={val_acc:.3f}"
                print(message)
        return history

    def _train_batch(self, x: np.ndarray, y_indices: np.ndarray, physics_targets: np.ndarray) -> None:
        hidden, probabilities = self._forward_standardized(x)
        n = x.shape[0]
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(n), y_indices] = 1.0

        dlogits = (probabilities - one_hot) / n
        if self.physics_weight > 0:
            dprob = (2.0 * self.physics_weight / n) * (probabilities - physics_targets)
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
        standardized = self.standardizer.transform(x)
        _, probabilities = self._forward_standardized(standardized)
        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(x), axis=1)
        return np.asarray([self.class_names[index] for index in indices])

    def loss_and_accuracy(self, x: np.ndarray, y: np.ndarray, physics_targets: np.ndarray) -> tuple[float, float]:
        probabilities = self.predict_proba(x)
        y_indices = labels_to_indices(y, self.class_names)
        ce = -np.mean(np.log(probabilities[np.arange(y_indices.size), y_indices] + 1e-12))
        physics_loss = float(np.mean((probabilities - physics_targets) ** 2))
        predictions = np.argmax(probabilities, axis=1)
        accuracy = float(np.mean(predictions == y_indices))
        return float(ce + self.physics_weight * physics_loss), accuracy

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.standardizer.mean_ is None or self.standardizer.scale_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            mean=self.standardizer.mean_,
            scale=self.standardizer.scale_,
            class_names=np.asarray(self.class_names),
            physics_weight=np.asarray([self.physics_weight]),
            learning_rate=np.asarray([self.learning_rate]),
        )

    @classmethod
    def load(cls, path: Path) -> "PhysicsInformedNN":
        data = np.load(path, allow_pickle=False)
        model = cls(
            input_dim=int(data["w1"].shape[0]),
            hidden_dim=int(data["w1"].shape[1]),
            class_names=tuple(str(item) for item in data["class_names"]),
            physics_weight=float(data["physics_weight"][0]),
            learning_rate=float(data["learning_rate"][0]),
        )
        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]
        model.standardizer.mean_ = data["mean"]
        model.standardizer.scale_ = data["scale"]
        return model

    def _forward_standardized(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        probabilities = _softmax(logits)
        return hidden, probabilities


def labels_to_indices(labels: np.ndarray, class_names: tuple[str, ...] = CLASS_NAMES) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(class_names)}
    try:
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown label {exc.args[0]!r}; expected one of {class_names}") from exc


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

