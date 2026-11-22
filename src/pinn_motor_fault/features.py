"""Physics-informed signal feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from tqdm import tqdm

from .paderborn import parse_shaft_frequency_hz

CLASS_NAMES = ("healthy", "inner_race", "outer_race", "rolling_element")

@dataclass(frozen=True)
class BearingGeometry6203:
    """Paderborn 6203 bearing geometry.

    Values are retained for reproducibility and are derived from the standard
    rolling-element bearing characteristic-frequency equations.
    """

    rolling_elements: int = 8
    rolling_element_diameter_mm: float = 6.75
    pitch_diameter_mm: float = 28.55
    contact_angle_deg: float = 0.0

    source: str = "Paderborn University Bearing Data Center; 6203 geometry"


@dataclass(frozen=True)
class BearingPhysics:
    """Corrected 6203 characteristic-frequency ratios and spectral settings."""

    geometry: BearingGeometry6203 = BearingGeometry6203()
    sample_rate_hz: float = 64_000.0
    harmonic_count: int = 5
    tolerance_hz: float | None = None

    @property
    def bpfo_ratio(self) -> float:
        theta = math.radians(self.geometry.contact_angle_deg)
        ratio = self.geometry.rolling_element_diameter_mm / self.geometry.pitch_diameter_mm
        return (self.geometry.rolling_elements / 2.0) * (1.0 - ratio * math.cos(theta))

    @property
    def bpfi_ratio(self) -> float:
        theta = math.radians(self.geometry.contact_angle_deg)
        ratio = self.geometry.rolling_element_diameter_mm / self.geometry.pitch_diameter_mm
        return (self.geometry.rolling_elements / 2.0) * (1.0 + ratio * math.cos(theta))

    @property
    def bsf_ratio(self) -> float:
        theta = math.radians(self.geometry.contact_angle_deg)
        d = self.geometry.rolling_element_diameter_mm
        D = self.geometry.pitch_diameter_mm
        ratio = d / D
        return (D / (2.0 * d)) * (1.0 - (ratio * math.cos(theta)) ** 2)

    @property
    def ftf_ratio(self) -> float:
        theta = math.radians(self.geometry.contact_angle_deg)
        ratio = self.geometry.rolling_element_diameter_mm / self.geometry.pitch_diameter_mm
        return 0.5 * (1.0 - ratio * math.cos(theta))

    def characteristic_frequencies(self, shaft_hz: float) -> dict[str, float]:
        if shaft_hz <= 0:
            raise ValueError("shaft_hz must be positive.")
        return {
            "outer_race": self.bpfo_ratio * shaft_hz,
            "inner_race": self.bpfi_ratio * shaft_hz,
            "rolling_element": self.bsf_ratio * shaft_hz,
            "shaft": shaft_hz,
            "ftf": self.ftf_ratio * shaft_hz,
        }

    def frequency_resolution_hz(self, window_size: int) -> float:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        return float(self.sample_rate_hz / window_size)

    def diagnostic_tolerance_hz(self, window_size: int) -> float:
        if self.tolerance_hz is not None:
            return float(self.tolerance_hz)
        return float(self.frequency_resolution_hz(window_size))


@dataclass
class FeatureBatch:
    features: np.ndarray
    physics_targets: np.ndarray
    feature_names: list[str]
    preprocessing: str = "mean_center"
    feature_group: str = "all"
    sample_rate_hz: float = 64_000.0
    shaft_frequencies_hz: np.ndarray | None = None
    geometry: dict[str, float | int | str] | None = None


class PhysicsFeatureExtractor:
    """Computes engineered signal features AND `physics_targets` for the
    physics-consistency loss term used by `model.PhysicsInformedNN`
    (`use_physics_loss=True`). This is Mechanism A in this project's two
    "physics-informed" mechanisms -- see model.py's module note and
    paper_features.py's module docstring for Mechanism B (the reference
    paper's feature-selection algorithm), which is unrelated to this class.
    """

    def __init__(
        self,
        physics: BearingPhysics | None = None,
        preprocessing: str = "mean_center",
        feature_group: str = "all",
    ) -> None:
        # Physics targets are generated from each raw window independently of
        # the classifier feature group.
        self.physics = physics or BearingPhysics()
        if preprocessing not in {"mean_center", "mean_center_standardize"}:
            raise ValueError("preprocessing must be 'mean_center' or 'mean_center_standardize'.")
        if feature_group not in {"generic", "diagnostic", "all"}:
            raise ValueError("feature_group must be 'generic', 'diagnostic', or 'all'.")
        self.preprocessing = preprocessing
        self.feature_group = feature_group

    def transform(self, windows: np.ndarray, sources: list[str] | None = None) -> FeatureBatch:
        # Convert raw windows into feature vectors, physics targets, and feature names.
        windows = np.asarray(windows, dtype=np.float64)
        if windows.ndim != 2:
            raise ValueError("windows must have shape (n_windows, n_samples).")
        feature_rows: list[np.ndarray] = []
        physics_rows: list[np.ndarray] = []
        names: list[str] | None = None
        for index, window in enumerate(tqdm(windows, desc='Extracting features')):
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
            preprocessing=self.preprocessing,
            feature_group=self.feature_group,
            sample_rate_hz=float(self.physics.sample_rate_hz),
            shaft_frequencies_hz=np.asarray([parse_shaft_frequency_hz(source) for source in (sources or [])], dtype=np.float64) if sources else None,
            geometry={
                "bearing": "6203",
                "n": self.physics.geometry.rolling_elements,
                "d_mm": self.physics.geometry.rolling_element_diameter_mm,
                "D_mm": self.physics.geometry.pitch_diameter_mm,
                "contact_angle_deg": self.physics.geometry.contact_angle_deg,
                "source": self.physics.geometry.source,
                "bpfo_ratio": self.physics.bpfo_ratio,
                "bpfi_ratio": self.physics.bpfi_ratio,
                "bsf_ratio": self.physics.bsf_ratio,
                "ftf_ratio": self.physics.ftf_ratio,
            },
        )

    def _one_window(self, window: np.ndarray, shaft_hz: float) -> tuple[np.ndarray, list[str], np.ndarray]:
        # Extract time-domain, spectral, and envelope-based diagnostic features for a single window.
        x = np.asarray(window, dtype=np.float64).reshape(-1)
        x = x - np.mean(x)
        if self.preprocessing == "mean_center_standardize":
            x = x / (np.std(x) + 1e-12)

        time_values, time_names = _time_features(x)
        spectrum, freqs = _single_sided_spectrum(x, self.physics.sample_rate_hz)
        spectral_values, spectral_names = _spectral_features(spectrum, freqs)
        diagnostic_values = np.asarray([], dtype=np.float64)
        diagnostic_names: list[str] = []

        if self.feature_group in {"diagnostic", "all"}:
            envelope = _analytic_envelope(x)
            envelope_spectrum, envelope_freqs = _single_sided_spectrum(
                envelope - np.mean(envelope), self.physics.sample_rate_hz
            )
            diagnostic_values, diagnostic_names = _diagnostic_features(
                envelope_spectrum,
                envelope_freqs,
                shaft_hz,
                self.physics,
                tolerance_hz=self.physics.diagnostic_tolerance_hz(x.size),
            )
        physics_target = physics_target_from_vibration_window(
            window,
            shaft_hz,
            self.physics,
        )

        generic_values = np.concatenate([time_values, spectral_values])
        generic_names = time_names + spectral_names
        if self.feature_group == "generic":
            values, names = generic_values, generic_names
        elif self.feature_group == "diagnostic":
            values, names = diagnostic_values, diagnostic_names
        else:
            values = np.concatenate([generic_values, diagnostic_values])
            names = generic_names + diagnostic_names
        return values, names, physics_target


def _time_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract time-domain features from a normalized vibration/current signal.

    `x` has already been mean-centered and divided by its own std in
    `_one_window()` before reaching here -- feature names below follow a
    `featuretype_feature` convention (all prefixed `time_`), with a further
    `normalized_` qualifier where that matters (e.g. `time_normalized_mean`
    is the mean of the already-centered signal, not the raw window's mean).
    """
    # Calculate statistical features
    abs_x = np.abs(x)
    rms = math.sqrt(float(np.mean(x * x)) + 1e-12)
    peak = float(np.max(abs_x))
    centered = x - np.mean(x)
    std = float(np.std(centered) + 1e-12)
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    mean_abs = float(np.mean(abs_x) + 1e-12)
    # Additional features
    zero_crossings = float(np.sum(np.diff(np.sign(x)) != 0))
    p = np.abs(x) / (np.sum(np.abs(x)) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))

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
            zero_crossings,
            entropy,
        ],
        dtype=np.float64,
    )
    names = ["time_normalized_mean", "time_normalized_std", "time_normalized_rms", "time_normalized_peak_amplitude",
             "time_normalized_peak_to_peak", "time_skewness", "time_kurtosis",
             "time_crest_factor", "time_shape_factor", "time_impulse_factor",
             "time_zero_crossings", "time_entropy"]
    return values, names


def _single_sided_spectrum(x: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    # Compute the positive-frequency power spectrum using a Hanning window.
    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    return spectrum + 1e-18, freqs


def _spectral_features(power: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    # Compute spectral summary statistics and normalized band energies.
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
    names = [
        "spectral_centroid", "spectral_bandwidth", "spectral_entropy",
        "spectral_band_energy_0_10pct_nyquist", "spectral_band_energy_10_30pct_nyquist",
        "spectral_band_energy_30_60pct_nyquist", "spectral_band_energy_60_100pct_nyquist",
    ]
    return values, names


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    # Compute the analytic signal envelope via FFT-based Hilbert transform.
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


def physics_target_from_vibration_window(
    window: np.ndarray,
    shaft_hz: float,
    physics: BearingPhysics | None = None,
) -> np.ndarray:
    """Create a label-independent soft target from envelope-spectrum energy.

    The healthy component is energy outside the bearing-fault harmonic bands;
    each fault component is energy assigned to its nearest BPFO, BPFI, or BSF
    harmonic within one diagnostic FFT-bin tolerance. These are consistency
    targets, not ground-truth labels or calibrated posterior probabilities.
    """
    physics = physics or BearingPhysics()
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("A vibration window must contain at least two samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("A vibration window must contain only finite values.")
    x = x - np.mean(x)
    envelope = _analytic_envelope(x)
    power, freqs = _single_sided_spectrum(
        envelope - np.mean(envelope),
        physics.sample_rate_hz,
    )
    band_mask = (freqs >= 5.0) & (freqs <= 2000.0)
    band_indices = np.flatnonzero(band_mask)
    if band_indices.size == 0:
        raise ValueError("The vibration window has no envelope-spectrum bins in 5-2000 Hz.")

    characteristic = physics.characteristic_frequencies(shaft_hz)
    fault_names = ("inner_race", "outer_race", "rolling_element")
    centers: list[tuple[float, str]] = []
    for fault_name in fault_names:
        base_frequency = characteristic[fault_name]
        for harmonic in range(1, physics.harmonic_count + 1):
            center = base_frequency * harmonic
            if 5.0 <= center <= 2000.0:
                centers.append((center, fault_name))

    assigned = np.zeros(freqs.size, dtype=bool)
    fault_energy = {fault_name: 0.0 for fault_name in fault_names}
    tolerance = physics.diagnostic_tolerance_hz(x.size)
    if centers:
        center_frequencies = np.asarray([item[0] for item in centers])
        center_classes = [item[1] for item in centers]
        distances = np.abs(freqs[band_indices, None] - center_frequencies[None, :])
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(band_indices.size), nearest]
        matched_indices = band_indices[nearest_distance <= tolerance]
        matched_classes = [center_classes[index] for index in nearest[nearest_distance <= tolerance]]
        for bin_index, fault_name in zip(matched_indices, matched_classes):
            fault_energy[fault_name] += float(power[bin_index])
            assigned[bin_index] = True

    healthy_energy = float(np.sum(power[band_indices[~assigned[band_indices]]]))
    raw_target = np.asarray(
        [healthy_energy, *(fault_energy[name] for name in fault_names)],
        dtype=np.float64,
    )
    target_sum = float(np.sum(raw_target))
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float64)
    return raw_target / target_sum


def _diagnostic_features(
    envelope_power: np.ndarray,
    freqs: np.ndarray,
    shaft_hz: float,
    physics: BearingPhysics,
    tolerance_hz: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    # Build diagnostic features around bearing characteristic harmonics.
    total = float(np.sum(envelope_power[(freqs >= 5.0) & (freqs <= 2000.0)]) + 1e-18)
    characteristic = physics.characteristic_frequencies(shaft_hz)
    values: list[float] = []
    names: list[str] = []
    class_scores: list[float] = []
    for label in ("outer_race", "inner_race", "rolling_element"):
        base_frequency = characteristic[label]
        harmonic_energies = []
        for harmonic in range(1, physics.harmonic_count + 1):
            center = base_frequency * harmonic
            tolerance = tolerance_hz if tolerance_hz is not None else physics.tolerance_hz
            if tolerance is None:
                resolution = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
                tolerance = resolution
            mask = np.abs(freqs - center) <= tolerance
            energy = float(np.sum(envelope_power[mask]) / total)
            harmonic_energies.append(energy)
            values.append(energy)
            names.append(f"diagnostic_{label}_envelope_h{harmonic}_energy_ratio")
        class_scores.append(float(np.sum(harmonic_energies)))
    max_fault_score = max(class_scores, default=0.0)
    values.extend([shaft_hz, max_fault_score])
    names.extend(["diagnostic_shaft_frequency_hz", "diagnostic_max_fault_class_envelope_energy_ratio"])
    return np.asarray(values, dtype=np.float64), names


def select_feature_group(features: np.ndarray, feature_names: list[str], feature_group: str) -> tuple[np.ndarray, list[str]]:
    """Select generic or diagnostic columns using names rather than indices."""
    if feature_group not in {"generic", "diagnostic", "all"}:
        raise ValueError("feature_group must be 'generic', 'diagnostic', or 'all'.")
    if feature_group == "all":
        return features, list(feature_names)
    if feature_group == "generic":
        indices = [i for i, name in enumerate(feature_names) if name.startswith(("time_", "spectral_"))]
    else:
        indices = [i for i, name in enumerate(feature_names) if name.startswith("diagnostic_")]
    if not indices:
        raise ValueError(f"No {feature_group} features found.")
    return features[:, indices], [feature_names[i] for i in indices]


class Standardizer:
    def __init__(self) -> None:
        # Simple standard scaler for feature normalization.
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
