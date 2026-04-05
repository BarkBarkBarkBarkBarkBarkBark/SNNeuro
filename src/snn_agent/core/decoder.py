# AGENT-HINT: Control signal decoder for closed-loop BCI experiments.
# PURPOSE: Converts L1 (or DEC) spike activity → scalar control signal.
# STRATEGIES: "discrete" (1/0 per step), "ttl" (fixed-width pulse),
#             "rate" (sliding window), "population" (leaky integrator),
#             "trigger" (decaying pulse).
# CONFIDENCE: Sliding-window average of DN spike activity.
# CONFIG: DecoderConfig in config.py (strategy, window_ms, threshold, ttl_*, etc.)
# SEE ALSO: template.py (L1 spikes), dec_layer.py (DEC spikes), app.py (UDP output)
"""
snn_agent.core.decoder — Control signal decoder for closed-loop experiments.

Converts L1 (TemplateLayer) spike activity into a scalar control signal
suitable for driving a stimulation controller or other experiment hardware.

Strategies:
    ``"discrete"``   — Clean 1/0 per time step: 1 if any spike + DN, else 0.
    ``"ttl"``        — Fixed-width TTL pulse: goes high for ``ttl_width_ms``
                       after a spike+DN event, then drops to 0.
    ``"rate"``       — Sliding-window spike rate → weighted sum.
    ``"population"`` — Leaky integrator; emits on threshold crossing.
    ``"trigger"``    — Decaying pulse on any L1 spike + DN active.

All strategies compute a **confidence** value (0–1) from recent DN activity.

Output: ``(control_value, confidence)``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from snn_agent.config import Config

__all__ = ["ControlDecoder"]


class ControlDecoder:
    """Converts L1 spike vectors + DN activity into a control signal.

    All strategies *always* return ``(control, confidence)`` — never ``None``.
    """

    def __init__(self, cfg: Config, n_l1: int) -> None:
        dec = cfg.decoder
        self.strategy = dec.strategy
        self.n = n_l1

        # Sampling rate — this constructor receives the effective
        # (post-decimation) config from pipeline.py, so sampling_rate_hz
        # is already the correct downstream rate.
        self._fs = float(cfg.sampling_rate_hz)

        # Control weights — default is *ones* (not 1/n), because WTA means
        # exactly one neuron fires per step.  Normalising by n makes each
        # spike contribution vanishingly small.
        if dec.weights is None:
            self.weights = np.ones(n_l1, dtype=np.float64)
        else:
            self.weights = np.array(dec.weights, dtype=np.float64)

        # Rate strategy — sliding window.
        # Stores per-step weighted dot-product floats rather than full spike arrays,
        # enabling an O(1) running sum that avoids the O(window × n) re-accumulation.
        win_samples = max(2, int(dec.window_ms * 1e-3 * self._fs))
        self._rate_window = win_samples
        self._spike_buf: deque[float] = deque(maxlen=win_samples)
        self._max_rate_hz = float(dec.max_rate_hz)
        self._rate_weighted_sum: float = 0.0  # O(1) running weighted spike count

        # Population strategy — leaky integrator
        tau_samples = dec.leaky_tau_ms * 1e-3 * self._fs
        self._pop_decay = np.exp(-1.0 / max(tau_samples, 1.0))
        self._pop_integrator = 0.0
        self._pop_threshold = dec.threshold

        # Trigger strategy — decaying pulse
        self._trigger_val = 0.0
        self._trigger_decay = self._pop_decay  # reuse same time constant

        # TTL strategy — fixed-width digital pulse
        self._ttl_width_samples = max(1, int(dec.ttl_width_ms * 1e-3 * self._fs))
        self._ttl_high = float(dec.ttl_high)
        self._ttl_countdown: int = 0  # samples remaining in current pulse

        # DN confidence — sliding window with O(1) running sum
        dn_win = int(dec.dn_confidence_window_ms * 1e-3 * self._fs)
        self._dn_buf: deque[bool] = deque(maxlen=max(dn_win, 1))
        self._dn_sum: int = 0  # running sum — updated incrementally, avoids sum(deque) each step

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
        # O(1) running confidence: track evicted item before appending
        evicted = self._dn_buf[0] if len(self._dn_buf) == self._dn_buf.maxlen else False
        self._dn_buf.append(dn_spike)
        self._dn_sum += int(dn_spike) - int(evicted)
        confidence = self._dn_sum / len(self._dn_buf)

        if self.strategy == "discrete":
            return self._step_discrete(l1_spikes, dn_spike, confidence)
        elif self.strategy == "ttl":
            return self._step_ttl(l1_spikes, dn_spike, confidence)
        elif self.strategy == "rate":
            return self._step_rate(l1_spikes, confidence)
        elif self.strategy == "population":
            return self._step_population(l1_spikes, confidence)
        elif self.strategy == "trigger":
            return self._step_trigger(l1_spikes, dn_spike, confidence)
        else:
            raise ValueError(f"Unknown ctrl_strategy: {self.strategy!r}")

    # ── strategies ────────────────────────────────────────────────────
    def _step_discrete(
        self, spikes: np.ndarray, dn_spike: bool, confidence: float
    ) -> tuple[float, float]:
        """Clean 1/0: fires exactly on the step a spike+DN event occurs."""
        if spikes.any() and dn_spike:
            return (1.0, confidence)
        return (0.0, confidence)

    def _step_ttl(
        self, spikes: np.ndarray, dn_spike: bool, confidence: float
    ) -> tuple[float, float]:
        """Fixed-width TTL pulse: goes high for ttl_width_ms, then low."""
        # New event restarts the pulse (extends if already high)
        if spikes.any() and dn_spike:
            self._ttl_countdown = self._ttl_width_samples
        if self._ttl_countdown > 0:
            self._ttl_countdown -= 1
            return (self._ttl_high, confidence)
        return (0.0, confidence)

    def _step_rate(
        self, spikes: np.ndarray, confidence: float
    ) -> tuple[float, float]:
        # O(1) per step: compute the weighted dot product for the new entry,
        # subtract the evicted entry's stored float, no full re-accumulation.
        new_val = float(np.dot(self.weights, spikes))
        if len(self._spike_buf) == self._spike_buf.maxlen:
            self._rate_weighted_sum -= self._spike_buf[0]
        self._rate_weighted_sum += new_val
        self._spike_buf.append(new_val)

        buf_len = len(self._spike_buf)
        if buf_len < 2:
            return (0.0, confidence)

        # Convert to Hz then normalise to [-1, 1]
        window_sec = buf_len / self._fs
        rate_hz = self._rate_weighted_sum / max(window_sec, 1e-10)
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
