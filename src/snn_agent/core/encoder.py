# AGENT-HINT: Temporal receptive field spike encoder.
# PURPOSE: Converts scalar electrode samples → boolean afferent vector via
#          amplitude-bin activation + shift-register delay taps.
# CALIBRATION: First 8000 samples → MAD noise estimate → bin geometry.
#   n_afferents is ONLY known after calibration. Pipeline is two-phase because of this.
# CONFIG: EncoderConfig in config.py (overlap, dvm_factor, step_size, window_depth)
# SEE ALSO: preprocessor.py (upstream), attention.py + template.py (downstream)
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

        # In-place amplitude bin activation — zero heap allocations
        np.subtract(self.centers, sample, out=self._abs_diff)
        np.abs(self._abs_diff, out=self._abs_diff)
        np.less_equal(self._abs_diff, self.dvm, out=self._active_bool)

        # Shift register: slide history left, insert new column on the right
        reg = self._shift_reg_2d                 # [n_centers, step_size*twindow]
        reg[:, :-1] = reg[:, 1:]                 # in-place left shift (no alloc)
        reg[:, -1]  = self._active_bool          # newest sample at rightmost column

        # For step_size>1: fill subsampled output buffer
        if self.step_size > 1:
            np.copyto(self._aff_out_2d, reg[:, :: self.step_size])
        # For step_size==1: _aff_out IS reg.ravel() — already up to date (zero-copy)

        return self._aff_out  # persistent view — no allocation, no copy

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

        nc = self.n_centers

        # 2-D shift register [n_centers, step_size * twindow] — C-contiguous.
        # Column 0 = oldest sample, column -1 = newest.
        self._shift_reg_2d = np.zeros((nc, self.step_size * self.twindow), dtype=bool)

        # _aff_out is a LIVE VIEW of the shift register for step_size=1
        # (no copy ever needed — direct memory access into the register).
        # For step_size>1 it’s a pre-allocated flat buffer filled by copyto.
        if self.step_size == 1:
            self._aff_out = self._shift_reg_2d.ravel()       # zero-copy persistent view
        else:
            self._aff_out = np.zeros(self.n_afferents, dtype=bool)  # pre-alloc copy target
            self._aff_out_2d = self._aff_out.reshape(nc, self.twindow)  # view for copyto

        # Pre-allocated scratch buffers for amplitude-bin computation (zero allocations per step)
        self._abs_diff    = np.empty(nc, dtype=np.float64)
        self._active_bool = np.empty(nc, dtype=bool)

        # Active indices cache — updated every step, read by sparse downstream layers
        self._active_indices: np.ndarray = np.empty(0, dtype=np.intp)

        self._abs_buf.clear()
        self._calibrated = True
