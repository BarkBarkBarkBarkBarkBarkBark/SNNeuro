# AGENT-HINT: Front-end signal conditioning.
# PURPOSE: Causal bandpass (300–6000 Hz) + decimation (÷4). Streaming-safe.
# OUTPUT: effective_fs = sampling_rate_hz / decimation_factor (e.g. 80k → 20k)
# OPTIMISED: step_batch() processes chunks via vectorized sosfilt; step() wraps
#            a pre-allocated 1-element array to avoid per-sample allocation.
# CONFIG: PreprocessConfig in config.py (bandpass_lo_hz, bandpass_hi_hz, etc.)
# SEE ALSO: encoder.py (downstream), pipeline.py (effective_fs propagation)
"""
snn_agent.core.preprocessor — Causal bandpass filter + decimation.

Streaming-safe: maintains IIR filter state across sample-by-sample calls.

**Optimised for Jetson:** ``step()`` reuses a pre-allocated 1-element array
to avoid per-sample ``np.array([x])`` allocation.  ``step_batch()`` accepts
a chunk of N raw samples and filters + decimates them in a single vectorised
``sosfilt`` call, amortising Python/C crossing overhead.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from snn_agent.config import Config

__all__ = ["Preprocessor"]


class Preprocessor:
    """
    Optional front-end: bandpass filter then decimate.

    Bandpass uses second-order sections (SOS) applied via ``sosfilt`` with
    carried state, so it is causal and streaming-safe.

    Decimation simply keeps every *N*-th sample (after the anti-alias bandpass
    has already removed energy above Nyquist / N).

    Usage::

        pp = Preprocessor(cfg)
        for raw_sample in stream:
            for sample in pp.step(raw_sample):
                encoder.step(sample)

        # Or batch mode (faster):
        for chunk in stream_chunks:
            decimated = pp.step_batch(chunk)
            for sample in decimated:
                encoder.step(sample)
    """

    def __init__(self, cfg: Config) -> None:
        fs = cfg.sampling_rate_hz
        pp = cfg.preprocess

        # ── Bandpass ──────────────────────────────────────────────────
        self.do_bandpass = pp.enable_bandpass
        if self.do_bandpass:
            self._sos = butter(
                pp.bandpass_order,
                [pp.bandpass_lo_hz, pp.bandpass_hi_hz],
                btype="band",
                fs=fs,
                output="sos",
            )
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
            # Pre-allocated 1-element buffer for step() (avoids np.array([x]))
            self._x1 = np.empty(1, dtype=np.float64)

        # ── Decimation ────────────────────────────────────────────────
        self.do_decimate = pp.enable_decimation
        self.dec_factor = pp.decimation_factor
        self._dec_count = 0

        # Effective sample rate after decimation
        self.effective_fs: int = fs // self.dec_factor if self.do_decimate else fs

    def step(self, sample: float) -> list[float]:
        """
        Ingest one raw sample.

        Returns ``[filtered_sample]`` or ``[]`` if decimated away.
        """
        x = sample

        if self.do_bandpass:
            self._x1[0] = x
            out, self._zi = sosfilt(self._sos, self._x1, zi=self._zi)
            x = float(out[0])

        if self.do_decimate:
            self._dec_count += 1
            if self._dec_count < self.dec_factor:
                return []
            self._dec_count = 0

        return [x]

<<<<<<< HEAD
    def step_batch(self, samples: np.ndarray) -> np.ndarray:
        """
        Process a chunk of raw samples in one vectorised call.

        Parameters
        ----------
        samples : np.ndarray
            1-D float array of raw electrode samples.

        Returns
        -------
        np.ndarray
            1-D float array of filtered, decimated samples.
            May be shorter than input (decimation) or empty.
        """
        x = np.asarray(samples, dtype=np.float64)
=======
    def step_chunk(self, samples: np.ndarray) -> np.ndarray:
        """
        Ingest a chunk of raw samples for one channel.

        Parameters
        ----------
        samples : ndarray [N]
            Contiguous block of raw samples.

        Returns
        -------
        ndarray [N_dec]
            Bandpass-filtered, decimated output.  Length is
            ``ceil((N - phase) / dec_factor)`` where *phase* is the
            current decimation counter.
        """
        x = samples.astype(np.float64)
>>>>>>> 316d4a9 (running fast parallel, still low observability and no output from the final layer)

        if self.do_bandpass:
            x, self._zi = sosfilt(self._sos, x, zi=self._zi)

        if self.do_decimate:
<<<<<<< HEAD
            # Continue the decimation counter from where step() left off
            # Find indices that land on the decimation stride
            start = (self.dec_factor - self._dec_count) % self.dec_factor
            x = x[start::self.dec_factor]
            # Update counter for next call
            total = len(samples)
            self._dec_count = (self._dec_count + total) % self.dec_factor

        return x
=======
            return self.decimate_chunk(x)

        return x

    def decimate_chunk(self, filtered_1d: np.ndarray) -> np.ndarray:
        """Apply decimation only (``filtered_1d`` is already bandpassed if used)."""
        x = np.asarray(filtered_1d, dtype=np.float64)
        if not self.do_decimate:
            return x
        remaining = self.dec_factor - self._dec_count
        if remaining > len(x):
            self._dec_count += len(x)
            return np.empty(0, dtype=np.float64)
        keep_idx = np.arange(remaining - 1, len(x), self.dec_factor)
        self._dec_count = (self._dec_count + len(x)) % self.dec_factor
        return x[keep_idx]
>>>>>>> 316d4a9 (running fast parallel, still low observability and no output from the final layer)
