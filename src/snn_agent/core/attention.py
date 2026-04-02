# AGENT-HINT: Attention neuron (DN) — excitatory energy/outlier detector.
# PURPOSE: Single LIF neuron with synaptic depression. Fires when signal energy
#          exceeds noise floor. Its spike excites L1 neurons and boosts decoder confidence.
# COUNTERPART: noise_gate.py is the inhibitory parallel — suppresses noise.
# CONFIG: DNConfig in config.py (threshold_factor, depression_tau, depression_frac)
# NUMBA:  Hot-path inner loop JIT-compiled for Jetson Orin Nano performance.
# SEE ALSO: template.py (consumes dn_spike), decoder.py (confidence gating)
"""
snn_agent.core.attention — Attention neuron (DN) with synaptic depression.

Single LIF neuron (no learnable weights) that detects when signal energy
exceeds the noise floor.  Provides excitatory boost for the template layer
and gating / confidence for the control decoder.

**Numba-accelerated:** The inner step function is JIT-compiled with
``@numba.njit`` to eliminate Python interpreter overhead in the hot loop.
"""

from __future__ import annotations

import numpy as np
import numba

from snn_agent.config import Config

__all__ = ["AttentionNeuron"]


# ── Numba-compiled hot path ──────────────────────────────────────────────────
@numba.njit(cache=True, fastmath=True)
def _dn_step_inner(
    afferents: np.ndarray,      # bool flat array
    v: float,                   # membrane potential
    p_rel: np.ndarray,          # release probabilities [n_afferents]
    last_pre: np.ndarray,       # last pre-spike time [n_afferents]
    t: int,                     # current timestep
    decay: float,               # membrane decay factor
    tau_d: float,               # depression time constant
    f_d: float,                 # depression fraction
    threshold: float,           # firing threshold
    reset_potential: float,     # post-fire reset V
    exp_lut: np.ndarray,        # pre-computed exp(-dt/tau_d) LUT
) -> tuple[float, bool]:
    """
    Single-step DN update — runs entirely in compiled machine code.

    Returns (new_v, fired).
    """
    # Find active afferents
    n_active = 0
    for i in range(len(afferents)):
        if afferents[i]:
            n_active += 1

    if n_active == 0:
        return v * decay, False

    # Process active afferents: recover pRel, compute contributions
    contribution_sum = 0.0
    lut_max = len(exp_lut) - 1
    for i in range(len(afferents)):
        if not afferents[i]:
            continue
        dt = t - last_pre[i]
        if dt < 0:
            dt = 0
        if dt > lut_max:
            dt = lut_max
        exp_val = exp_lut[dt]

        # Recover pRel
        p_rel[i] = 1.0 - (1.0 - p_rel[i]) * exp_val
        contribution_sum += p_rel[i]

        # Depress pRel
        p_rel[i] *= (1.0 - f_d)
        last_pre[i] = t

    # Membrane update
    v = v * decay + contribution_sum
    fired = v >= threshold
    if fired:
        v = reset_potential

    return v, fired


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

        self.tau_d = float(dn.depression_tau)
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
        self.v, fired = _dn_step_inner(
            afferents,
            self.v,
            self.p_rel,
            self.last_pre,
            self.t,
            self.decay,
            self.tau_d,
            self.f_d,
            self.threshold,
            self.reset_potential,
            self._exp_td,
        )
        return fired
