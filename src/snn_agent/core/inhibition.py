# AGENT-HINT: Global post-spike inhibition module.
# PURPOSE: After any L1 neuron fires, suppress all L1 input for a blanking
#          period (default 5 ms) UNLESS the input current exceeds a strength
#          threshold. Mirrors cortical refractory blanking and the L2_wiFactor
#          lateral inhibition from the MATLAB Output_Layer.
# CONFIG:  InhibitionConfig in config.py (duration_ms, strength_threshold)
# WIRING:  Called by TemplateLayer.step() as a pre-step gate on input current.
# SEE ALSO: template.py (integration), config.py (InhibitionConfig),
#           docs/reference/MB2018-ANNet/Output_Layer/ (MATLAB L2 inhibition)
"""
snn_agent.core.inhibition — Global post-spike inhibitory blanking.

After any L1 neuron fires, all L1 input current is suppressed for a
configurable blanking window (default 5 ms) unless the raw input current
exceeds a strength threshold, allowing strong signals to break through.

This implements a biologically-inspired refractory/inhibitory mechanism
analogous to cortical interneuron-mediated blanking.
"""

from __future__ import annotations

from snn_agent.config import Config

__all__ = ["GlobalInhibitor"]


class GlobalInhibitor:
    """
    Global post-spike inhibition gate.

    When any L1 neuron fires, a countdown timer activates. While active,
    all L1 input currents are zeroed *unless* the total current magnitude
    exceeds ``strength_threshold``.

    Parameters are drawn from ``Config.inhibition``.

    Usage in TemplateLayer.step()::

        current = afferents @ W + dn_boost
        current = inhibitor.gate(current, any_l1_fired_last_step)
    """

    def __init__(self, cfg: Config) -> None:
        inh = cfg.inhibition
        effective_fs = cfg.effective_fs()

        # Convert ms → samples
        self.blanking_samples = max(
            1, int(inh.duration_ms * 1e-3 * effective_fs)
        )
        self.strength_threshold = inh.strength_threshold

        # State
        self._countdown: int = 0
        self.active: bool = False

    def gate(self, current_magnitude: float, any_fired: bool) -> float:
        """
        Apply inhibitory gating.

        Parameters
        ----------
        current_magnitude : float
            The total input current magnitude (pre-gating).
        any_fired : bool
            Whether any L1 neuron fired on the *previous* timestep.

        Returns
        -------
        float
            Multiplicative suppression factor: 1.0 (pass) or 0.0 (suppress).
            Returns 1.0 if current exceeds strength threshold even during
            blanking (strong-signal bypass).
        """
        # Trigger blanking on any spike
        if any_fired:
            self._countdown = self.blanking_samples

        if self._countdown > 0:
            self._countdown -= 1
            self.active = True

            # Strong-signal bypass: let through if current is large enough
            if current_magnitude >= self.strength_threshold:
                return 1.0

            return 0.0

        self.active = False
        return 1.0

    def reset(self) -> None:
        """Reset inhibitor state."""
        self._countdown = 0
        self.active = False
