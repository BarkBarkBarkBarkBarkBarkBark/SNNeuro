# AGENT-HINT: Spiking decoder layer — 16 neurons for putative unit identification.
# PURPOSE: Neuron 0 = DN-gated any-fire detector.  Neurons 1–15 learn via
#          competitive STDP to each represent one putative neural unit from L1.
# INPUTS:  l1_spikes [n_l1], dn_spike (bool).  All integration is DN-gated.
# OUTPUT:  dec_spikes [16] boolean + hex word (uint16).
# DELAYS:  Optional delay expansion (toggle via DECConfig.use_delays).
# CONFIG:  DECConfig in config.py.
# SEE ALSO: template.py (L1 upstream), decoder.py (control signal downstream),
#           config.py (DECConfig), pipeline.py (wiring).
"""
snn_agent.core.dec_layer — Spiking decoder layer for putative unit identification.

Architecture
~~~~~~~~~~~~

16 LIF neurons that receive L1 template-layer spikes and learn to produce
one spike per putative neural unit:

- **Neuron 0 ("any-fire")**: Hard-wired unit weights from all L1 neurons.
  Fires whenever *any* L1 neuron fires, provided the DN attention neuron
  is also active.  Threshold is kept very low.  No STDP — this neuron is
  a pure DN-gated presence detector.

- **Neurons 1–15 ("unit neurons")**: Random initial weights from L1,
  competitive WTA + asymmetric STDP.  Each neuron converges to represent
  a cluster of L1 responses corresponding to one putative neural unit.
  Only integrates during DN-active timesteps (noise-driven L1 spikes are
  ignored).

Delay expansion (optional, ``DECConfig.use_delays=True``) adds a shift
register that gives each DEC neuron a spatio-temporal receptive field
over the recent history of L1 firing — similar to the MATLAB ANNet L2
delay taps (51 × 50 µs).

Output is a boolean vector ``[16]`` plus a convenience hex property
encoding which neurons fired as a uint16 bitmask.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import snntorch as snn

from snn_agent.config import Config

__all__ = ["DECLayer"]


class DECLayer:
    """16-neuron spiking decoder layer with DN gating and optional delays.

    Parameters
    ----------
    cfg : Config
        Full configuration (reads ``cfg.dec``).
    n_l1 : int
        Number of L1 template neurons (input dimensionality).
    """

    N_DEC = 16  # total neurons (1 any-fire + 15 learned)

    def __init__(self, cfg: Config, n_l1: int) -> None:
        dc = cfg.dec
        self.n_l1 = n_l1
        n = self.N_DEC
        n_unit = n - 1  # learned neurons (1–15)

        # ── Delay expansion ───────────────────────────────────────────
        self.use_delays = dc.use_delays
        self.n_taps = dc.n_delay_taps if dc.use_delays else 1
        # Shift register: each entry is a bool array [n_l1]
        self._delay_buf: deque[np.ndarray] = deque(
            maxlen=self.n_taps,
        )
        for _ in range(self.n_taps):
            self._delay_buf.append(np.zeros(n_l1, dtype=bool))

        # Effective input size: n_l1 × n_taps (flattened)
        self.n_input = n_l1 * self.n_taps

        # ── Time constant / LIF parameters ────────────────────────────
        tm = dc.tm_samples
        beta = float(np.exp(-1.0 / tm))

        # ── Thresholds ────────────────────────────────────────────────
        self.any_fire_threshold = dc.any_fire_threshold
        self.unit_threshold = float(dc.unit_threshold_factor * n_l1)

        # ── Neuron 0: any-fire (fixed weights = 1) ───────────────────
        # Implemented separately (not in WTA group) so it doesn't
        # compete with the unit neurons.
        self.lif_any = snn.Leaky(
            beta=beta,
            threshold=self.any_fire_threshold,
            reset_mechanism="zero",
        )
        self.mem_any = self.lif_any.init_leaky()
        # Fixed weight: 1.0 for every delayed L1 input
        self._w_any = torch.ones(self.n_input, dtype=torch.float32)

        # ── Neurons 1–15: learned unit neurons (WTA) ─────────────────
        self.lif_unit = snn.Leaky(
            beta=beta,
            threshold=self.unit_threshold,
            inhibition=True,  # WTA among unit neurons
            reset_mechanism="zero",
        )
        self.mem_unit = self.lif_unit.init_leaky()

        # Learnable weights [n_input, n_unit]
        rng = np.random.default_rng(seed=17)
        w_np = rng.uniform(
            dc.init_w_min, dc.init_w_max, (self.n_input, n_unit)
        ).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # Lateral inhibition scaling
        self.wi_factor = dc.wi_factor

        # ── STDP ──────────────────────────────────────────────────────
        self.ltp = dc.stdp_ltp
        self.ltp_win = dc.stdp_ltp_window
        self.ltd = dc.stdp_ltd
        self.w_lo = dc.w_lo
        self.w_hi = dc.w_hi
        self.freeze = dc.freeze_stdp

        # ── State tracking ────────────────────────────────────────────
        self.n = n
        self.n_unit = n_unit
        self.t: int = 0
        self.refractory = dc.refractory_samples
        self.last_post_spike = np.full(n_unit, -9999, dtype=np.int64)
        self.last_pre_spike = np.full(self.n_input, -9999, dtype=np.int64)

        # Pre-allocated buffers
        self._x_buf = torch.zeros(self.n_input, dtype=torch.float32)

        # Output cache
        self._last_hex: int = 0

    # ── public API ────────────────────────────────────────────────────
    @property
    def hex_output(self) -> int:
        """Last output as uint16 bitmask (bit 0 = neuron 0, etc.)."""
        return self._last_hex

    def step(self, l1_spikes: np.ndarray, dn_spike: bool) -> np.ndarray:
        """Advance one timestep.

        Parameters
        ----------
        l1_spikes : np.ndarray
            Boolean array ``[n_l1]`` — which L1 neurons fired.
        dn_spike : bool
            Whether the attention neuron (DN) fired this step.

        Returns
        -------
        np.ndarray
            Boolean array ``[16]`` — which DEC neurons fired.
        """
        self.t += 1
        out = np.zeros(self.N_DEC, dtype=bool)

        # ── Update delay buffer ───────────────────────────────────────
        self._delay_buf.append(l1_spikes.astype(bool).copy())

        # ── DN gate: skip all integration if DN is not active ─────────
        if not dn_spike:
            self._last_hex = 0
            return out

        # ── Build flattened input from delay buffer ───────────────────
        if self.use_delays:
            flat_input = np.concatenate(list(self._delay_buf))
        else:
            flat_input = l1_spikes.astype(bool)

        # Track pre-spike times
        active = np.flatnonzero(flat_input)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

        # Build input tensor
        self._x_buf.zero_()
        if len(active) > 0:
            self._x_buf[active] = 1.0

        # ── Neuron 0: any-fire ────────────────────────────────────────
        any_current = torch.dot(self._x_buf, self._w_any)
        with torch.no_grad():
            spk0, self.mem_any = self.lif_any(
                any_current.unsqueeze(0), self.mem_any.unsqueeze(0)
            )
            spk0 = spk0.squeeze(0)
            self.mem_any = self.mem_any.squeeze(0)
        out[0] = bool(spk0.item() > 0.5)

        # ── Neurons 1–15: learned unit neurons ────────────────────────
        current = self._x_buf @ self.W  # [n_unit]

        # Lateral inhibition scaling (amplify WTA competition)
        if self.wi_factor > 1.0:
            mean_c = current.mean()
            current = mean_c + (current - mean_c) * self.wi_factor

        with torch.no_grad():
            spk_u, self.mem_unit = self.lif_unit(
                current.unsqueeze(0), self.mem_unit.unsqueeze(0)
            )
            spk_u = spk_u.squeeze(0)
            self.mem_unit = self.mem_unit.squeeze(0)

        unit_spikes = spk_u.numpy().astype(bool)

        # Enforce refractory + STDP on winners
        for w in np.flatnonzero(unit_spikes):
            if (self.t - self.last_post_spike[w]) <= self.refractory:
                unit_spikes[w] = False
                continue
            self.last_post_spike[w] = self.t
            if not self.freeze:
                self._stdp(int(w))

        out[1:] = unit_spikes

        # ── Encode as hex bitmask ─────────────────────────────────────
        self._last_hex = int(sum(1 << i for i in range(self.N_DEC) if out[i]))

        return out

    # ── STDP ──────────────────────────────────────────────────────────
    def _stdp(self, winner: int) -> None:
        """Asymmetric Hebbian STDP on the winning unit neuron's column."""
        w = self.W[:, winner].numpy()

        # Global LTD
        w += self.ltd

        # LTP for recently active pre-synaptic inputs
        dt = self.t - self.last_pre_spike
        causal = (dt <= self.ltp_win) & (self.last_pre_spike > 0)
        w[causal] += self.ltp

        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = torch.from_numpy(w)
