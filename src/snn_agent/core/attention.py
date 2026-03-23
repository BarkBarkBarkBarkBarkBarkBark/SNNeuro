# AGENT-HINT: Attention neuron (DN) — excitatory energy/outlier detector.
# PURPOSE: Single LIF neuron with synaptic depression. Fires when signal energy
#          exceeds noise floor. Its spike excites L1 neurons and boosts decoder confidence.
# COUNTERPART: noise_gate.py is the inhibitory parallel — suppresses noise.
# CONFIG: DNConfig in config.py (threshold_factor, depression_tau, depression_frac)
# SEE ALSO: template.py (consumes dn_spike), decoder.py (confidence gating)
"""
snn_agent.core.attention — Attention neuron (DN) with synaptic depression.

Single LIF neuron (no learnable weights) that detects when signal energy
exceeds the noise floor.  Provides excitatory boost for the template layer
and gating / confidence for the control decoder.
"""

from __future__ import annotations

import numpy as np

from snn_agent.config import Config

__all__ = ["AttentionNeuron"]


class AttentionNeuron:
    """
    Single LIF neuron with short-term synaptic depression (pRel).

    Each afferent contributes its release probability *pRel* as the weight::

        Recovery:   pRel(t) = 1 − (1 − pRel_last) × exp(−Δt / τ_d)
        Depression: pRel(t) *= (1 − f_d)

    Fires when V ≥ threshold.  Soft reset (V set to ``reset_potential``).
    """

    def __init__(self, cfg: Config, n_afferents: int) -> None:
        dn = cfg.dn
        enc = cfg.encoder
        tm = dn.tm_samples

        self.threshold = (
            dn.threshold_factor
            * enc.overlap
            * enc.window_depth
            / (1.0 - np.exp(-1.0 / tm))
        )
        self.reset_potential = (
            dn.reset_potential_factor
            * (np.exp(1.0 / tm) - 1.0)
            * self.threshold
        )

        self.tau_d = dn.depression_tau
        self.f_d = dn.depression_frac
        self.decay = np.exp(-1.0 / tm)

        # State
        self.v: float = 0.0
        self.p_rel = np.ones(n_afferents, dtype=np.float64)
        self.last_pre = np.full(n_afferents, -9999, dtype=np.int64)
        self.t: int = 0

        # Pre-computed exp LUT for depression recovery
        max_dt = int(self.tau_d * 20)
        self._exp_td = np.exp(
            -np.arange(max_dt + 1, dtype=np.float64) / self.tau_d
        )

    def step(self, afferents: np.ndarray) -> bool:
        """
        Ingest the flat afferent vector (bool).

        Returns ``True`` if the attention neuron fires this timestep.
        """
        self.t += 1
        active = np.flatnonzero(afferents)

        if len(active) == 0:
            self.v *= self.decay
            return False

        # Update pRel for active afferents
        dt_arr = self.t - self.last_pre[active]
        dt_arr = np.clip(dt_arr, 0, len(self._exp_td) - 1)
        exp_vals = self._exp_td[dt_arr]

        self.p_rel[active] = 1.0 - (1.0 - self.p_rel[active]) * exp_vals
        contributions = self.p_rel[active].copy()
        self.p_rel[active] *= 1.0 - self.f_d
        self.last_pre[active] = self.t

        self.v = self.v * self.decay + contributions.sum()

        fired = self.v >= self.threshold
        if fired:
            self.v = self.reset_potential

        return bool(fired)
