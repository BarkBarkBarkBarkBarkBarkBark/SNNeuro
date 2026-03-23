# AGENT-HINT: Kalman-filter noise gate — parallel inhibitory pathway.
# PURPOSE: Runs alongside the AttentionNeuron (DN). While DN excites L1 when
#          outlier energy is detected, the NoiseGateNeuron INHIBITS L1 when
#          signal variance is within the noise floor (i.e., "this is just noise").
# CONFIG:  NoiseGateConfig in config.py (process_noise, measurement_noise,
#          inhibit_below_sd, suppression_factor)
# WIRING:  Called in pipeline loop (app.py / evaluate.py) after encoder.step(),
#          produces a suppression factor [0, 1] that scales L1 input current.
# SEE ALSO: attention.py (excitatory counterpart), template.py (current scaling),
#           config.py (NoiseGateConfig), pipeline.py (integration)
"""
snn_agent.core.noise_gate — Kalman-filter noise suppression gate.

Implements a 1-D Kalman filter that tracks signal variance in real time.
When the estimated standard deviation is close to the predicted noise
baseline (within ``inhibit_below_sd`` standard deviations of the Kalman
state), the gate emits a suppression factor that scales down L1 input,
reducing responses to background noise.

This is the inhibitory complement to the AttentionNeuron (DN):
  - DN says "something interesting is here" → excite L1
  - NoiseGate says "this is just noise" → suppress L1

Together they provide push–pull modulation of the template layer.
"""

from __future__ import annotations

import math

import numpy as np

from snn_agent.config import Config

__all__ = ["NoiseGateNeuron"]


class NoiseGateNeuron:
    """
    1-D Kalman filter tracking signal variance for noise gating.

    State model:
        - State ``x``: estimated signal variance (σ²)
        - Measurement: squared sample value (instantaneous power)
        - Process noise ``Q``: how fast we expect variance to change
        - Measurement noise ``R``: observation uncertainty

    When √x (estimated σ) is within ``inhibit_below_sd`` of the calibrated
    noise floor, the gate outputs a suppression factor < 1.0, which scales
    down L1 input current.

    The gate needs the encoder's noise estimate (σ_noise) to establish
    a baseline. This is available after encoder calibration.
    """

    def __init__(self, cfg: Config, noise_sigma: float) -> None:
        ng = cfg.noise_gate

        # Kalman parameters
        self.Q = ng.process_noise       # process noise covariance
        self.R = ng.measurement_noise   # measurement noise covariance
        self.inhibit_below_sd = ng.inhibit_below_sd
        self.suppression_factor = ng.suppression_factor

        # Noise baseline from encoder calibration (MAD estimate)
        self.noise_variance = noise_sigma ** 2
        self.noise_sigma = noise_sigma

        # Kalman state
        self.x = self.noise_variance    # state estimate (variance)
        self.P = 1.0                    # estimate covariance

        # Exponential moving average for smoother gating
        self._ema_alpha = 0.05
        self._ema_suppression = 1.0

        # Public state for monitoring
        self.estimated_sigma: float = noise_sigma
        self.is_suppressing: bool = False

    def step(self, sample: float) -> float:
        """
        Ingest one pre-processed sample, return suppression factor.

        Parameters
        ----------
        sample : float
            The bandpass-filtered, decimated electrode sample.

        Returns
        -------
        float
            Suppression factor in ``[suppression_factor, 1.0]``.
            1.0 = no suppression (signal looks interesting).
            suppression_factor = maximum suppression (signal looks like noise).
        """
        # Measurement: instantaneous squared amplitude
        z = sample * sample

        # ── Kalman predict ────────────────────────────────────────────
        # State prediction (random walk model: x_pred = x)
        x_pred = self.x
        P_pred = self.P + self.Q

        # ── Kalman update ─────────────────────────────────────────────
        # Innovation
        y = z - x_pred
        S = P_pred + self.R  # innovation covariance
        K = P_pred / S       # Kalman gain

        self.x = x_pred + K * y
        self.P = (1.0 - K) * P_pred

        # Clamp state to positive (variance can't be negative)
        if self.x < 1e-12:
            self.x = 1e-12

        self.estimated_sigma = math.sqrt(self.x)

        # ── Gating decision ──────────────────────────────────────────
        # How many noise-floor SDs away is the current estimate?
        if self.noise_sigma > 0:
            deviation_ratio = self.estimated_sigma / self.noise_sigma
        else:
            deviation_ratio = 1.0

        # If estimated σ is close to noise floor → suppress
        if deviation_ratio < self.inhibit_below_sd:
            # Linear interpolation: closer to noise = more suppression
            t = deviation_ratio / self.inhibit_below_sd
            raw_factor = self.suppression_factor + t * (1.0 - self.suppression_factor)
        else:
            raw_factor = 1.0

        # Smooth with EMA to avoid rapid toggling
        self._ema_suppression = (
            self._ema_alpha * raw_factor
            + (1.0 - self._ema_alpha) * self._ema_suppression
        )

        self.is_suppressing = self._ema_suppression < 0.95
        return self._ema_suppression

    def reset(self, noise_sigma: float | None = None) -> None:
        """Reset Kalman state, optionally with new noise baseline."""
        if noise_sigma is not None:
            self.noise_sigma = noise_sigma
            self.noise_variance = noise_sigma ** 2
        self.x = self.noise_variance
        self.P = 1.0
        self._ema_suppression = 1.0
        self.is_suppressing = False
