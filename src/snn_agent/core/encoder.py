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

        # Circular shift register (replaces np.roll for O(1) writes)
        self._ring: np.ndarray | None = None   # flat bool ring buffer
        self._ring_len: int = 0                  # n_centers * step_size * twindow
        self._ring_head: int = 0                 # write pointer
        # Pre-built read indices for subsample extraction (avoids reshape/slice per step)
        self._read_indices: np.ndarray | None = None
        # Pre-allocated output buffer (avoids ravel allocation per step)
        self._out_buf: np.ndarray | None = None

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

    # ── post-calibration reshape ───────────────────────────────────────
    def pad_to_n_centers(self, target_nc: int) -> None:
        """
        Extend the encoder's amplitude-bin grid to exactly *target_nc* centres.

        Extra centres are placed at ``sig_max + dvm × 1000`` — well outside
        the calibrated signal range — so they are **never activated** during
        inference (``|sample − dummy| ≫ dvm``).  All internal arrays are
        resized in-place so the encoder's ``step()`` continues to work
        correctly without any caller changes.

        Call this **after** calibration (``is_calibrated`` must be True) and
        **before** building any downstream layers.  No-op when
        ``target_nc <= self.n_centers``.
        """
        assert self._calibrated, "pad_to_n_centers requires a calibrated encoder"
        old_nc = self.n_centers
        if target_nc <= old_nc:
            return

        extra = target_nc - old_nc

        # Dummy centres: place them far outside the real signal range so they
        # can never satisfy |sample − centre| <= dvm.
        dummy_val = self._sig_max + self.dvm * 1000.0
        dummy_centers = np.full(extra, dummy_val, dtype=np.float64)
        self.centers = np.concatenate([self.centers, dummy_centers])

        # Extend the shift register with zero rows for the new dummy centres.
        reg_width = self.step_size * self.twindow
        new_reg = np.zeros((target_nc, reg_width), dtype=bool)
        new_reg[:old_nc, :] = self._shift_reg_2d
        self._shift_reg_2d = new_reg

        # Update counts.
        self.n_centers = target_nc
        self.n_afferents = target_nc * self.twindow

        # Rebuild per-step scratch buffers (size changed).
        self._abs_diff    = np.empty(target_nc, dtype=np.float64)
        self._active_bool = np.empty(target_nc, dtype=bool)

        # Rebuild afferent output buffer / live view.
        if self.step_size == 1:
            self._aff_out = self._shift_reg_2d.ravel()          # zero-copy live view
        else:
            self._aff_out    = np.zeros(self.n_afferents, dtype=bool)
            self._aff_out_2d = self._aff_out.reshape(target_nc, self.twindow)

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
