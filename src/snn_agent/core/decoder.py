# AGENT-HINT: Control signal decoder for closed-loop BCI experiments.
# PURPOSE: Converts L1 (or L2 if enabled) spike activity → scalar control signal.
# STRATEGIES: "rate" (sliding window), "population" (leaky integrator), "trigger" (binary).
# CONFIDENCE: Sliding-window average of DN spike activity.
# CONFIG: DecoderConfig in config.py (strategy, window_ms, threshold, etc.)
# SEE ALSO: template.py (L1 spikes), output_layer.py (L2 spikes), app.py (UDP output)
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
    """Converts L1 spike vectors + DN activity into a control signal.

    All strategies *always* return ``(control, confidence)`` — never ``None``.
    The ``control`` value is a continuous signal in [-1, 1] suitable for both
    visualisation and closed-loop output.
    """

    def __init__(self, cfg: Config, n_l1: int) -> None:
        dec = cfg.decoder
        self.strategy = dec.strategy
        self.n = n_l1

        # Effective sampling rate (post-decimation)
        self._fs = float(cfg.effective_fs())

        # Control weights — default is *ones* (not 1/n), because WTA means
        # exactly one neuron fires per step.  Normalising by n makes each
        # spike contribution vanishingly small.
        if dec.weights is None:
            self.weights = np.ones(n_l1, dtype=np.float64)
        else:
            self.weights = np.array(dec.weights, dtype=np.float64)

        # Rate strategy — sliding window
        win_samples = max(2, int(dec.window_ms * 1e-3 * self._fs))
        self._rate_window = win_samples
        self._spike_buf: deque[np.ndarray] = deque(maxlen=win_samples)
        self._max_rate_hz = float(dec.max_rate_hz)

        # Population strategy — leaky integrator
        tau_samples = dec.leaky_tau_ms * 1e-3 * self._fs
        self._pop_decay = np.exp(-1.0 / max(tau_samples, 1.0))
        self._pop_integrator = 0.0
        self._pop_threshold = dec.threshold

        # Trigger strategy — decaying pulse
        self._trigger_val = 0.0
        self._trigger_decay = self._pop_decay  # reuse same time constant

        # DN confidence — sliding window
        dn_win = int(dec.dn_confidence_window_ms * 1e-3 * self._fs)
        self._dn_buf: deque[bool] = deque(maxlen=max(dn_win, 1))

        self.t: int = 0

    # ── public API ────────────────────────────────────────────────────
    def step(
        self, l1_spikes: np.ndarray, dn_spike: bool
    ) -> tuple[float, float]:
        """
        Ingest one timestep.

        Always returns ``(control_value, confidence)``.
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
    ) -> tuple[float, float]:
        self._spike_buf.append(spikes.copy())
        buf_len = len(self._spike_buf)
        if buf_len < 2:
            return (0.0, confidence)

        counts = np.zeros(self.n, dtype=np.float64)
        for s in self._spike_buf:
            counts += s.astype(np.float64)

        # Convert to Hz then normalise to [-1, 1]
        window_sec = buf_len / self._fs
        weighted_count = float(np.dot(self.weights, counts))
        rate_hz = weighted_count / max(window_sec, 1e-10)
        control = float(np.clip(rate_hz / self._max_rate_hz, -1.0, 1.0))
        return (control, confidence)

    def _step_population(
        self, spikes: np.ndarray, confidence: float
    ) -> tuple[float, float]:
        self._pop_integrator *= self._pop_decay
        if np.any(spikes):
            self._pop_integrator += float(
                np.dot(self.weights, spikes.astype(np.float64))
            )

        # Always return current integrator progress toward threshold —
        # 0 = idle, 1 = just crossing threshold.
        control = float(
            np.clip(self._pop_integrator / self._pop_threshold, -1.0, 1.0)
        )

        # Reset on threshold crossing (emission event)
        if self._pop_integrator >= self._pop_threshold:
            self._pop_integrator = 0.0

        return (control, confidence)

    def _step_trigger(
        self, spikes: np.ndarray, dn_spike: bool, confidence: float
    ) -> tuple[float, float]:
        self._trigger_val *= self._trigger_decay
        if np.any(spikes) and dn_spike:
            self._trigger_val = 1.0
        return (self._trigger_val, confidence)
