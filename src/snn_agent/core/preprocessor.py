# AGENT-HINT: Front-end signal conditioning.
# PURPOSE: Causal bandpass (300–6000 Hz) + decimation (÷4). Streaming-safe.
# OUTPUT: effective_fs = sampling_rate_hz / decimation_factor (e.g. 80k → 20k)
# CONFIG: PreprocessConfig in config.py (bandpass_lo_hz, bandpass_hi_hz, etc.)
# SEE ALSO: encoder.py (downstream), pipeline.py (effective_fs propagation)
"""
snn_agent.core.preprocessor — Causal bandpass filter + decimation.

Streaming-safe: maintains IIR filter state across sample-by-sample calls.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from snn_agent.config import Config

__all__ = ["Preprocessor"]


class Preprocessor:
    """
    Optional front-end: bandpass filter then decimate.

    Bandpass uses second-order sections (SOS) applied sample-by-sample via
    ``sosfilt`` with carried state, so it is causal and streaming-safe.

    Decimation simply keeps every *N*-th sample (after the anti-alias bandpass
    has already removed energy above Nyquist / N).

    Usage::

        pp = Preprocessor(cfg)
        for raw_sample in stream:
            for sample in pp.step(raw_sample):
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
            out, self._zi = sosfilt(self._sos, np.array([x]), zi=self._zi)
            x = float(out[0])

        if self.do_decimate:
            self._dec_count += 1
            if self._dec_count < self.dec_factor:
                return []
            self._dec_count = 0

        return [x]
