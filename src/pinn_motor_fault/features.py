"""Physics-informed signal feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .paderborn import parse_shaft_frequency_hz

CLASS_NAMES = ("healthy", "inner_race", "outer_race", "rolling_element")


@dataclass(frozen=True)
class BearingPhysics:
    sample_rate_hz: float = 64_000.0
    bpfo_ratio: float = 3.5848
    bpfi_ratio: float = 5.4152
    bsf_ratio: float = 2.3570
    harmonic_count: int = 5
    tolerance_hz: float = 4.0

    def characteristic_frequencies(self, shaft_hz: float) -> dict[str, float]:
        return {
            "outer_race": self.bpfo_ratio * shaft_hz,
            "inner_race": self.bpfi_ratio * shaft_hz,
            "rolling_element": self.bsf_ratio * shaft_hz,
        }


@dataclass
class FeatureBatch:
    features: np.ndarray
    physics_targets: np.ndarray
    feature_names: list[str]


class PhysicsFeatureExtractor:
    def __init__(self, physics: BearingPhysics | None = None) -> None:
        self.physics = physics or BearingPhysics()

    def transform(self, windows: np.ndarray, sources: list[str] | None = None) -> FeatureBatch:
        feature_rows: list[np.ndarray] = []
        physics_rows: list[np.ndarray] = []
        names: list[str] | None = None
        for index, window in enumerate(windows):
            source = sources[index] if sources and index < len(sources) else None
            shaft_hz = parse_shaft_frequency_hz(source)
            features, feature_names, physics_target = self._one_window(window, shaft_hz)
            feature_rows.append(features)
            physics_rows.append(physics_target)
            names = feature_names
        return FeatureBatch(
            features=np.vstack(feature_rows).astype(np.float64),
            physics_targets=np.vstack(physics_rows).astype(np.float64),
            feature_names=names or [],
        )

    def _one_window(self, window: np.ndarray, shaft_hz: float) -> tuple[np.ndarray, list[str], np.ndarray]:
        x = np.asarray(window, dtype=np.float64).reshape(-1)
        x = x - np.mean(x)
        std = np.std(x) + 1e-12
        x = x / std

        time_values, time_names = _time_features(x)
        spectrum, freqs = _single_sided_spectrum(x, self.physics.sample_rate_hz)
        spectral_values, spectral_names = _spectral_features(spectrum, freqs)
        envelope = _analytic_envelope(x)
        envelope_spectrum, envelope_freqs = _single_sided_spectrum(envelope - np.mean(envelope), self.physics.sample_rate_hz)
        diagnostic_values, diagnostic_names, physics_target = _diagnostic_features(
            envelope_spectrum,
            envelope_freqs,
            shaft_hz,
            self.physics,
        )

        values = np.concatenate([time_values, spectral_values, diagnostic_values])
        names = time_names + spectral_names + diagnostic_names
        return values, names, physics_target


def _time_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract time-domain features with enhanced preprocessing:
    - Normalization (zero mean, unit variance)
    - Skewness and kurtosis
    - Peak/RMS ratios
    - Statistical moments
    """
    abs_x = np.abs(x)
    rms = math.sqrt(float(np.mean(x * x)) + 1e-12)
    peak = float(np.max(abs_x))
    centered = x - np.mean(x)
    std = float(np.std(centered) + 1e-12)
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    mean_abs = float(np.mean(abs_x) + 1e-12)
    values = np.asarray(
        [
            float(np.mean(x)),
            float(np.std(x)),
            rms,
            peak,
            float(np.ptp(x)),
            skew,
            kurtosis,
            peak / rms,
            rms / mean_abs,
            peak / mean_abs,
        ],
        dtype=np.float64,
    )
    names = ["mean", "std", "rms", "peak", "ptp", "skew", "kurtosis", "crest_factor", "shape_factor", "impulse_factor"]
    return values, names


def _single_sided_spectrum(x: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    return spectrum + 1e-18, freqs


def _spectral_features(power: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    total = float(np.sum(power) + 1e-18)
    probabilities = power / total
    centroid = float(np.sum(freqs * probabilities))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * probabilities)))
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-18)) / np.log(probabilities.size))
    nyquist = float(freqs[-1])
    bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
    band_energy = []
    for low, high in bands:
        mask = (freqs >= low * nyquist) & (freqs < high * nyquist)
        band_energy.append(float(np.sum(power[mask]) / total))
    values = np.asarray([centroid, bandwidth, entropy, *band_energy], dtype=np.float64)
    names = ["spectral_centroid", "spectral_bandwidth", "spectral_entropy", "band_0_10", "band_10_30", "band_30_60", "band_60_100"]
    return values, names


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(x.size)
    if x.size % 2 == 0:
        multiplier[0] = 1.0
        multiplier[x.size // 2] = 1.0
        multiplier[1 : x.size // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (x.size + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    return np.abs(analytic)


def _diagnostic_features(
    envelope_power: np.ndarray,
    freqs: np.ndarray,
    shaft_hz: float,
    physics: BearingPhysics,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    total = float(np.sum(envelope_power[(freqs >= 5.0) & (freqs <= 2000.0)]) + 1e-18)
    characteristic = physics.characteristic_frequencies(shaft_hz)
    values: list[float] = []
    names: list[str] = []
    class_scores = {"healthy": 0.0, "inner_race": 0.0, "outer_race": 0.0, "rolling_element": 0.0}
    for label in ("outer_race", "inner_race", "rolling_element"):
        base_frequency = characteristic[label]
        harmonic_energies = []
        for harmonic in range(1, physics.harmonic_count + 1):
            center = base_frequency * harmonic
            mask = np.abs(freqs - center) <= physics.tolerance_hz
            energy = float(np.sum(envelope_power[mask]) / total)
            harmonic_energies.append(energy)
            values.append(energy)
            names.append(f"{label}_h{harmonic}_energy")
        class_scores[label] = float(np.sum(harmonic_energies))

    max_fault_score = max(class_scores["inner_race"], class_scores["outer_race"], class_scores["rolling_element"])
    class_scores["healthy"] = max(0.0, 0.05 - max_fault_score)
    raw = np.asarray([class_scores[name] for name in CLASS_NAMES], dtype=np.float64)
    physics_target = (raw + 1e-6) / float(np.sum(raw + 1e-6))
    values.extend([shaft_hz, max_fault_score])
    names.extend(["shaft_frequency_hz", "max_fault_harmonic_energy"])
    return np.asarray(values, dtype=np.float64), names, physics_target


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

