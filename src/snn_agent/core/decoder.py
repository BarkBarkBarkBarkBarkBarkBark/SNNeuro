"""
snn_agent.core.decoder — Control signal decoder for closed-loop experiments.

Converts L1 (TemplateLayer) spike activity into a scalar control signal
suitable for driving a stimulation controller or other experiment hardware.

Strategies:
    ``"rate"``       — Sliding-window spike rate → weighted sum.
    ``"population"`` — Leaky integrator; emits on threshold crossing.
    ``"trigger"``    — Binary pulse on any L1 spike + DN active.

All strategies compute a **confidence** value (0–1) from recent DN activity.

Output: ``(control_value, confidence)`` or ``None``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from snn_agent.config import Config

__all__ = ["ControlDecoder"]


class ControlDecoder:
    """Converts L1 spike vectors + DN activity into a control signal."""

    def __init__(self, cfg: Config, n_l1: int) -> None:
        dec = cfg.decoder
        self.strategy = dec.strategy
        self.n = n_l1

        # Control weights
        if dec.weights is None:
            self.weights = np.ones(n_l1, dtype=np.float64) / n_l1
        else:
            self.weights = np.array(dec.weights, dtype=np.float64)

        # Rate strategy — sliding window
        sr = cfg.sampling_rate_hz
        win_samples = int(dec.window_ms * 1e-3 * sr)
        self._rate_window = win_samples
        self._spike_buf: deque[np.ndarray] = deque(maxlen=win_samples)

        # Population strategy — leaky integrator
        tau_samples = dec.leaky_tau_ms * 1e-3 * sr
        self._pop_decay = np.exp(-1.0 / max(tau_samples, 1.0))
        self._pop_integrator = 0.0
        self._pop_threshold = dec.threshold

        # DN confidence — sliding window
        dn_win = int(dec.dn_confidence_window_ms * 1e-3 * sr)
        self._dn_buf: deque[bool] = deque(maxlen=max(dn_win, 1))

        self.t: int = 0

    # ── public API ────────────────────────────────────────────────────
    def step(
        self, l1_spikes: np.ndarray, dn_spike: bool
    ) -> tuple[float, float] | None:
        """
        Ingest one timestep.

        Returns ``(control_value, confidence)`` or ``None``.
        """
        self.t += 1
        self._dn_buf.append(dn_spike)
        confidence = sum(self._dn_buf) / len(self._dn_buf)

        if self.strategy == "rate":
            return self._step_rate(l1_spikes, confidence)
        elif self.strategy == "population":
            return self._step_population(l1_spikes, confidence)
        elif self.strategy == "trigger":
            return self._step_trigger(l1_spikes, dn_spike, confidence)
        else:
            raise ValueError(f"Unknown ctrl_strategy: {self.strategy!r}")

    # ── strategies ────────────────────────────────────────────────────
    def _step_rate(
        self, spikes: np.ndarray, confidence: float
    ) -> tuple[float, float] | None:
        self._spike_buf.append(spikes.copy())
        if len(self._spike_buf) < self._rate_window:
            return None

        counts = np.zeros(self.n, dtype=np.float64)
        for s in self._spike_buf:
            counts += s.astype(np.float64)

        rates = counts / self._rate_window
        raw = float(np.dot(self.weights, rates))
        control = float(np.clip(raw, -1.0, 1.0))
        return (control, confidence)

    def _step_population(
        self, spikes: np.ndarray, confidence: float
    ) -> tuple[float, float] | None:
        self._pop_integrator *= self._pop_decay
        if np.any(spikes):
            self._pop_integrator += float(
                np.dot(self.weights, spikes.astype(np.float64))
            )
        if self._pop_integrator >= self._pop_threshold:
            control = float(np.clip(self._pop_integrator, -1.0, 1.0))
            self._pop_integrator = 0.0
            return (control, confidence)
        return None

    def _step_trigger(
        self, spikes: np.ndarray, dn_spike: bool, confidence: float
    ) -> tuple[float, float] | None:
        if np.any(spikes) and dn_spike:
            return (1.0, confidence)
        return None
