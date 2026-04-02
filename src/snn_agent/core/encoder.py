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

        # ── Circular buffer write (O(1) — no array copy) ─────────────
        ring = self._ring
        nc = self.n_centers
        head = self._ring_head

        # Activate amplitude bins: vectorised |centres − sample| ≤ dvm
        active = np.abs(self.centers - sample) <= self.dvm

        # Write activated bins at head position in the ring
        ring[head:head + nc] = active
        # Advance head (mod ring length)
        self._ring_head = (head + nc) % self._ring_len

        # ── Read subsampled delay taps via pre-built index array ──────
        # _read_indices is shape [n_afferents] and contains the ring
        # positions to read (offset by current head), built once at
        # calibration time.  We add the current head and wrap.
        idx = (self._read_indices + self._ring_head) % self._ring_len
        np.take(ring, idx, out=self._out_buf)
        return self._out_buf

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

        # ── Circular ring buffer (replaces old shift register) ────────
        self._ring_len = self.n_centers * self.step_size * self.twindow
        self._ring = np.zeros(self._ring_len, dtype=bool)
        self._ring_head = 0

        # ── Pre-build read indices for subsampled delay taps ──────────
        # The original code did:
        #   reshaped = ring.reshape(step_size*twindow, n_centres).T
        #   out = reshaped[:, ::step_size]  # shape [n_centres, twindow]
        #   return out.ravel()
        #
        # We pre-compute the flat indices so each step is just a gather.
        # Row c, col t in the [n_centres × twindow] output corresponds to
        # flat ring position:  (t * step_size) * n_centres + c
        # Indices are relative to the *oldest* entry in the ring, which
        # at read time sits at ring_head (because we just advanced head
        # past the newest write).
        nc = self.n_centers
        ss = self.step_size
        tw = self.twindow
        indices = np.empty(self.n_afferents, dtype=np.intp)
        k = 0
        for c in range(nc):
            for t in range(tw):
                indices[k] = (t * ss) * nc + c
                k += 1
        self._read_indices = indices
        self._out_buf = np.empty(self.n_afferents, dtype=bool)

        self._abs_buf.clear()
        self._calibrated = True
