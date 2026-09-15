"""Physics-informed feature selection adapted from Li et al., EUSIPCO 2025.

Reference Algorithm 1:
    1. Compute averaged magnitude spectra for the relevant classes.
    2. Identify physics-defined frequency bins B.
    3. Z-score the averaged spectra.
    4. Compute d[i] = |zF[i] - zN[i]|.
    5. Select S = {i in B : d[i] >= tau}.

The reference paper is binary (nominal vs faulty). Paderborn contains
four bearing-condition classes, so this implementation uses a one-vs-rest
extension:

    for each class C:
        x_C     = mean spectrum of class C
        x_rest  = mean spectrum of all other classes
        d_C     = |z_C - z_rest|

The final score for a frequency bin is:

    d[i] = max_C d_C[i]

This preserves the mathematical structure of Algorithm 1 while extending
it to multiclass bearing diagnosis.

For the bearing dataset, the physics-defined frequency set B is built
around:
    - BPFO harmonics  (outer race)
    - BPFI harmonics  (inner race)
    - BSF harmonics   (rolling element)
    - shaft-frequency sidebands around those characteristic frequencies

This is a physics adaptation to the Paderborn bearing problem, not a
claim that the original paper used bearing frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import BearingPhysics


@dataclass
class PaperFeatureSelector:
    """Algorithm-1-style physics-informed frequency feature selector."""

    sample_rate_hz: float = 64_000.0

    # Algorithm 1 threshold tau.
    tau: float = 1.0

    # Number of harmonics of BPFO/BPFI/BSF to consider.
    harmonic_count: int = 5

    # Include f_c - f_shaft and f_c + f_shaft sidebands.
    include_sidebands: bool = True

    # Frequency tolerance is expressed in FFT bins.
    # 1.5 bins is appropriate for the 4096-sample windows used here.
    bin_radius: float = 1.5

    selected_indices: np.ndarray | None = None
    selected_freqs: np.ndarray | None = None
    diff_scores: np.ndarray | None = None

    # Useful for inspecting which class produced each selected feature.
    class_scores: dict[str, np.ndarray] | None = None

    def _magnitude_spectra(
        self,
        windows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute magnitude FFT for every signal window."""

        windows = np.asarray(windows, dtype=np.float64)

        if windows.ndim != 2:
            raise ValueError(
                f"Expected windows with shape (n_samples, n_points), "
                f"got {windows.shape}"
            )

        n = windows.shape[1]

        freqs = np.fft.rfftfreq(
            n,
            d=1.0 / self.sample_rate_hz,
        )

        spectra = np.empty(
            (windows.shape[0], freqs.size),
            dtype=np.float64,
        )

        for i, window in enumerate(windows):
            x = window - np.mean(window)

            # Same basic magnitude-spectrum representation as Algorithm 1.
            spectra[i] = np.abs(np.fft.rfft(x))

        return spectra, freqs

    def _physics_frequency_centers(
        self,
        shaft_hz: float,
        physics: BearingPhysics,
    ) -> list[float]:
        """Return bearing characteristic frequencies and sidebands."""

        characteristic = physics.characteristic_frequencies(shaft_hz)

        centers: list[float] = []

        for label in (
            "outer_race",
            "inner_race",
            "rolling_element",
        ):
            base_frequency = characteristic[label]

            for harmonic in range(1, self.harmonic_count + 1):

                fc = base_frequency * harmonic

                # Main characteristic-frequency harmonic.
                centers.append(fc)

                if self.include_sidebands:
                    # Bearing-fault sidebands around characteristic
                    # frequencies separated by shaft frequency.
                    centers.append(fc - shaft_hz)
                    centers.append(fc + shaft_hz)

        # Also retain the fundamental shaft frequency as a physical bin.
        centers.append(shaft_hz)

        return centers

    def _physics_bin_mask(
        self,
        freqs: np.ndarray,
        shaft_hz: float,
        physics: BearingPhysics,
    ) -> np.ndarray:
        """Construct the physics-defined frequency-bin set B."""

        centers = self._physics_frequency_centers(
            shaft_hz,
            physics,
        )

        # FFT frequency resolution.
        if freqs.size < 2:
            raise ValueError("FFT frequency vector is too short.")

        resolution_hz = float(freqs[1] - freqs[0])

        # Around 1.5 FFT bins.
        radius_hz = self.bin_radius * resolution_hz

        mask = np.zeros(freqs.shape, dtype=bool)

        for center in centers:
            # Ignore negative frequencies.
            if center < 0:
                continue

            # Ignore frequencies above Nyquist.
            if center > freqs[-1]:
                continue

            mask |= np.abs(freqs - center) <= radius_hz

        return mask

    @staticmethod
    def _zscore_spectrum(
        spectrum: np.ndarray,
    ) -> np.ndarray:
        """Z-score an averaged magnitude spectrum across frequency bins."""

        mean = float(np.mean(spectrum))
        std = float(np.std(spectrum))

        return (spectrum - mean) / (std + 1e-12)

    def fit(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        shaft_hz: float | None = None,
        physics: BearingPhysics | None = None,
    ) -> "PaperFeatureSelector":
        """Fit the Algorithm-1-style selector on TRAINING data only.

        Parameters
        ----------
        windows:
            Training signal windows.

        labels:
            Training labels.

        shaft_hz:
            Shaft rotational frequency in Hz.

            If omitted, 25 Hz is used, matching the default used by the
            project's Paderborn metadata parser.

        physics:
            BearingPhysics configuration.
        """

        windows = np.asarray(windows, dtype=np.float64)
        labels = np.asarray(labels)

        if windows.shape[0] != labels.shape[0]:
            raise ValueError(
                "windows and labels must contain the same number of samples."
            )

        if windows.shape[0] == 0:
            raise ValueError("Cannot fit PaperFeatureSelector on empty data.")

        physics = physics or BearingPhysics(
            sample_rate_hz=self.sample_rate_hz
        )

        if shaft_hz is None:
            shaft_hz = 25.0

        # ------------------------------------------------------------
        # Step 1: magnitude spectra
        # ------------------------------------------------------------
        spectra, freqs = self._magnitude_spectra(windows)

        # ------------------------------------------------------------
        # Step 2: identify physics-defined bins B
        # ------------------------------------------------------------
        physics_mask = self._physics_bin_mask(
            freqs,
            shaft_hz,
            physics,
        )

        physics_indices = np.flatnonzero(physics_mask)

        if physics_indices.size == 0:
            raise RuntimeError(
                "No physics-defined FFT bins were found. "
                "Check sample_rate_hz, shaft_hz and window size."
            )

        # ------------------------------------------------------------
        # Step 3: class-specific averaged spectra
        # ------------------------------------------------------------
        class_names = sorted(
            set(str(label) for label in labels)
        )

        if len(class_names) < 2:
            raise ValueError(
                "PaperFeatureSelector requires at least two classes."
            )

        class_scores: dict[str, np.ndarray] = {}

        # Maximum one-vs-rest z-score difference for every frequency bin.
        max_difference = np.zeros(
            freqs.size,
            dtype=np.float64,
        )

        for class_name in class_names:

            class_mask = np.asarray(
                [str(label) == class_name for label in labels]
            )

            rest_mask = ~class_mask

            if not np.any(class_mask):
                continue

            if not np.any(rest_mask):
                continue

            # Averaged magnitude spectrum for class C.
            x_class = np.mean(
                spectra[class_mask],
                axis=0,
            )

            # Averaged magnitude spectrum for all other classes.
            x_rest = np.mean(
                spectra[rest_mask],
                axis=0,
            )

            # --------------------------------------------------------
            # Step 4: z-scores
            # --------------------------------------------------------
            z_class = self._zscore_spectrum(x_class)
            z_rest = self._zscore_spectrum(x_rest)

            # --------------------------------------------------------
            # Step 5: d[i] = |z_class[i] - z_rest[i]|
            # --------------------------------------------------------
            difference = np.abs(
                z_class - z_rest
            )

            class_scores[class_name] = difference

            # Multiclass extension:
            # retain the strongest class-vs-rest difference.
            max_difference = np.maximum(
                max_difference,
                difference,
            )

        # ------------------------------------------------------------
        # Step 6: apply tau ONLY inside physics-defined B
        # ------------------------------------------------------------
        selected_mask = (
            physics_mask
            & (max_difference >= self.tau)
        )

        selected_indices = np.flatnonzero(
            selected_mask
        )

        # ------------------------------------------------------------
        # Safety check
        # ------------------------------------------------------------
        # Unlike the previous top-K implementation, we do NOT silently
        # select arbitrary FFT bins if tau produces zero features.
        if selected_indices.size == 0:
            max_physics_difference = float(
                np.max(max_difference[physics_indices])
            )

            raise RuntimeError(
                f"Algorithm 1 selected zero features at tau={self.tau}. "
                f"Maximum physics-bin difference was "
                f"{max_physics_difference:.4f}. "
                f"Try a lower tau, e.g. 0.5 or 0.75."
            )

        # Sort by frequency so feature ordering is deterministic.
        selected_indices = np.sort(selected_indices)

        self.selected_indices = selected_indices
        self.selected_freqs = freqs[selected_indices]
        self.diff_scores = max_difference[selected_indices]
        self.class_scores = class_scores

        return self

    def transform(
        self,
        windows: np.ndarray,
    ) -> np.ndarray:
        """Transform windows using the selected frequency bins."""

        if self.selected_indices is None:
            raise RuntimeError(
                "PaperFeatureSelector must be fit() before transform()."
            )

        spectra, _ = self._magnitude_spectra(windows)

        return spectra[:, self.selected_indices]

    def fit_transform(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        shaft_hz: float | None = None,
        physics: BearingPhysics | None = None,
    ) -> np.ndarray:
        """Fit selector and transform the supplied windows."""

        self.fit(
            windows,
            labels,
            shaft_hz=shaft_hz,
            physics=physics,
        )

        return self.transform(windows)

    def feature_names(self) -> list[str]:
        """Return human-readable names for selected features."""

        if self.selected_freqs is None:
            raise RuntimeError(
                "PaperFeatureSelector has not been fitted yet."
            )

        return [
            f"paper_freq_{frequency:.1f}hz"
            for frequency in self.selected_freqs
        ]