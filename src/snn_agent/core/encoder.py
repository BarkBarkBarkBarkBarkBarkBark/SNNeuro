"""
snn_agent.core.encoder — Temporal receptive field spike encoder.

Converts a stream of scalar electrode samples into a flat boolean afferent
vector via amplitude-bin activation + shift-register delay taps.
"""

from __future__ import annotations

import numpy as np

from snn_agent.config import Config

__all__ = ["SpikeEncoder"]


class SpikeEncoder:
    """
    Temporal receptive field encoder (ANNet-derived).

    Architecture:
        1. Estimate noise via running Median Absolute Deviation (MAD).
        2. Build a set of overlapping amplitude bins (centres).
        3. At each sample, activate the bins that bracket the amplitude.
        4. Feed activations into a shift register (``twindow`` delay taps,
           subsampled by ``step_size``).
        5. Flatten the ``[n_centres × twindow]`` matrix into a 1-D bool vector.
    """

    def __init__(self, cfg: Config) -> None:
        enc = cfg.encoder
        self.overlap = enc.overlap
        self.dvm_factor = enc.dvm_factor
        self.step_size = enc.step_size
        self.twindow = enc.window_depth
        self.init_samples = enc.noise_init_samples

        # State — populated after calibration
        self.centers: np.ndarray | None = None
        self.dvm: float = 0.0
        self.n_centers: int = 0
        self.n_afferents: int = 0

        # Shift register
        self._shift_reg: np.ndarray | None = None

        # Running noise / signal stats
        self._abs_buf: list[float] = []
        self._sig_min: float = 0.0
        self._sig_max: float = 0.0
        self._calibrated: bool = False
        self._sample_count: int = 0

    # ── public API ────────────────────────────────────────────────────
    def step(self, sample: float) -> np.ndarray:
        """
        Ingest one electrode sample.

        Returns a flat boolean array of shape ``(n_afferents,)``.
        During calibration returns an empty array.
        """
        self._sample_count += 1

        if not self._calibrated:
            self._abs_buf.append(abs(sample))
            if sample < self._sig_min:
                self._sig_min = sample
            if sample > self._sig_max:
                self._sig_max = sample
            if self._sample_count >= self.init_samples:
                self._calibrate()
            else:
                return np.zeros(0, dtype=bool)

        assert self.centers is not None and self._shift_reg is not None

        # Activate amplitude bins
        active = np.abs(self.centers - sample) <= self.dvm

        # Shift register: shift left, insert new at right
        self._shift_reg = np.roll(self._shift_reg, -self.n_centers)
        self._shift_reg[-self.n_centers:] = active

        # Subsample delay taps
        out = self._shift_reg.reshape(
            self.step_size * self.twindow, self.n_centers
        ).T
        current = out[:, :: self.step_size]  # [n_centers × twindow]
        return current.ravel()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── internals ─────────────────────────────────────────────────────
    def _calibrate(self) -> None:
        arr = np.array(self._abs_buf, dtype=np.float64)
        noise_est = float(np.median(arr) / 0.6745)
        self.dvm = self.dvm_factor * noise_est

        dvi = self.dvm * 2.0 / self.overlap
        self.centers = np.arange(
            self._sig_min - self.dvm,
            self._sig_max + self.dvm + dvi,
            dvi,
            dtype=np.float64,
        )
        self.n_centers = len(self.centers)
        self.n_afferents = self.n_centers * self.twindow

        reg_len = self.n_centers * self.step_size * self.twindow
        self._shift_reg = np.zeros(reg_len, dtype=bool)

        self._abs_buf.clear()
        self._calibrated = True
