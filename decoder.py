"""
decoder.py — Control signal decoder for closed-loop experiments.

Converts L1 (TemplateLayer) spike activity into a scalar control signal
suitable for driving a stimulation controller or other experiment hardware.

Three strategies (selected via cfg["ctrl_strategy"]):

    "rate"       — Sliding-window spike rate → weighted sum → control float.
                   Latency ≈ window size (default 5 ms).  Smooth output.

    "population" — Leaky integrator per L1 neuron.  On each spike,
                   integrator jumps by the neuron's control weight.
                   Emits when integrator crosses threshold.
                   Latency < 1 ms (spike-triggered).

    "trigger"    — Binary pulse whenever ANY L1 neuron fires AND the
                   attention neuron (DN) is active.  Simplest possible.
                   Latency < 0.5 ms.

All strategies also compute a *confidence* value (0–1) based on recent
DN (attention neuron) activity.  The experiment controller can suppress
stimulation when confidence is low.

Output tuple: (control_value: float, confidence: float) or None
"""

from collections import deque
import numpy as np


class ControlDecoder:
    """
    Converts L1 spike vectors + DN activity into a control signal.

    Usage:
        decoder = ControlDecoder(cfg, n_l1_neurons)
        ...
        result = decoder.step(l1_spikes, dn_spike)
        if result is not None:
            control_value, confidence = result
            send_udp(control_value, confidence)
    """

    def __init__(self, cfg: dict, n_l1: int):
        self.strategy = cfg["ctrl_strategy"]
        self.n = n_l1

        # Control weights: maps each L1 neuron to a contribution.
        # None → uniform (1/n).  Can be set after calibration.
        cw = cfg["ctrl_weights"]
        if cw is None:
            self.weights = np.ones(n_l1, dtype=np.float64) / n_l1
        else:
            self.weights = np.array(cw, dtype=np.float64)

        # Rate strategy — sliding window of spike vectors
        sr = cfg["sampling_rate_hz"]
        win_samples = int(cfg["ctrl_window_ms"] * 1e-3 * sr)
        self._rate_window = win_samples
        self._spike_buf: deque[np.ndarray] = deque(maxlen=win_samples)

        # Population strategy — leaky integrator
        tau_samples = cfg["ctrl_leaky_tau_ms"] * 1e-3 * sr
        self._pop_decay = np.exp(-1.0 / max(tau_samples, 1.0))
        self._pop_integrator = 0.0
        self._pop_threshold = cfg["ctrl_threshold"]

        # DN confidence — sliding window of bool
        dn_win = int(cfg["ctrl_dn_confidence_window_ms"] * 1e-3 * sr)
        self._dn_buf: deque[bool] = deque(maxlen=max(dn_win, 1))

        self.t = 0

    # ── public API ────────────────────────────────────────────────────────────
    def step(self, l1_spikes: np.ndarray, dn_spike: bool
             ) -> tuple[float, float] | None:
        """
        Ingest one timestep of L1 spikes and DN activity.

        Returns (control_value, confidence) or None if no control event.
        control_value ∈ [-1, 1],  confidence ∈ [0, 1].
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

    def reset(self) -> None:
        """Clear all internal state (e.g. between trials)."""
        self._spike_buf.clear()
        self._dn_buf.clear()
        self._pop_integrator = 0.0
        self.t = 0

    # ── strategies ────────────────────────────────────────────────────────────
    def _step_rate(self, spikes: np.ndarray, confidence: float
                   ) -> tuple[float, float] | None:
        """
        Sliding-window rate code.
        Always emits a control value (even if 0).
        """
        self._spike_buf.append(spikes.copy())

        if len(self._spike_buf) < self._rate_window:
            return None  # window not yet full

        # Count spikes per neuron over the window
        counts = np.zeros(self.n, dtype=np.float64)
        for s in self._spike_buf:
            counts += s.astype(np.float64)

        # Normalise to rate (spikes / window_length)
        rates = counts / self._rate_window

        # Weighted sum → control value, clamped to [-1, 1]
        raw = float(np.dot(self.weights, rates))
        control = float(np.clip(raw, -1.0, 1.0))

        return (control, confidence)

    def _step_population(self, spikes: np.ndarray, confidence: float
                         ) -> tuple[float, float] | None:
        """
        Leaky integrator.  Emits only when integrator ≥ threshold.
        """
        # Decay
        self._pop_integrator *= self._pop_decay

        # Accumulate weighted spike contributions
        if np.any(spikes):
            self._pop_integrator += float(np.dot(
                self.weights, spikes.astype(np.float64)))

        if self._pop_integrator >= self._pop_threshold:
            control = float(np.clip(self._pop_integrator, -1.0, 1.0))
            self._pop_integrator = 0.0  # reset after emission
            return (control, confidence)

        return None

    def _step_trigger(self, spikes: np.ndarray, dn_spike: bool,
                      confidence: float) -> tuple[float, float] | None:
        """
        Binary trigger: emit 1.0 when any L1 neuron fires AND DN is active.
        """
        if np.any(spikes) and dn_spike:
            return (1.0, confidence)
        return None
