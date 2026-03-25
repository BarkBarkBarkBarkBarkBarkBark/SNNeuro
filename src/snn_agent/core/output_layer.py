# AGENT-HINT: DEPRECATED — replaced by dec_layer.py.
# This file is retained for reference only.  It is NOT imported by pipeline.py
# and NOT wired into the active processing chain.  The DEC layer (dec_layer.py)
# provides DN-gated hierarchical competitive learning with 16 neurons instead.
# ORIGINAL: L2 classification / convergence layer from MATLAB ANNet Output_Layer.
# SEE ALSO: dec_layer.py (replacement), template.py (L1 upstream),
#           docs/reference/MB2018-ANNet/Output_Layer/ (MATLAB reference)
"""
snn_agent.core.output_layer — L2 classification layer with lateral inhibition.

Optional convergence layer that receives L1 template spikes and learns
to cluster them into discrete neural unit identities. Uses lateral
inhibition (winner-take-all with inhibitory weight factor) and
competitive STDP, mirroring the MATLAB ANNet Output_Layer.

This layer provides an additional opportunity for the network to learn
invariant representations: while L1 neurons become template-matchers
for specific waveform phases, L2 neurons learn to associate multiple
L1 responses into coherent unit identities.

Enabled only when ``Config.use_l2 = True``.
"""

from __future__ import annotations

import numpy as np
import torch
import snntorch as snn

from snn_agent.config import Config

__all__ = ["ClassificationLayer"]


class ClassificationLayer:
    """
    L2 classification layer: N_L2 LIF neurons with lateral inhibition + STDP.

    Input:  L1 spike vector ``[n_l1]``
    Output: L2 spike vector ``[n_l2]``

    Architecture:
        - Feed-forward weights ``W`` of shape ``[n_l1, n_l2]``
        - Lateral inhibition via snnTorch WTA (``inhibition=True``)
        - Additional lateral inhibitory weight factor ``wi_factor``
          that scales WTA suppression (stronger = more competitive)
        - Competitive asymmetric STDP (same rule as L1):
          global LTD on all afferents + causal LTP for recent pre-spikes
    """

    def __init__(self, cfg: Config, n_l1: int) -> None:
        l2 = cfg.l2
        n = l2.n_neurons
        tm = l2.tm_samples
        beta = float(np.exp(-1.0 / tm))

        self.n = n
        self.n_l1 = n_l1
        self.wi_factor = l2.wi_factor

        # Threshold: simpler than L1 since input is sparse spike counts
        # Base threshold scaled by expected L1 activity
        self.threshold = float(l2.threshold_factor * n_l1)

        # STDP
        self.ltp = l2.stdp_ltp
        self.ltp_win = l2.stdp_ltp_window
        self.ltd = l2.stdp_ltd
        self.w_lo = l2.w_lo
        self.w_hi = l2.w_hi
        self.freeze = l2.freeze_stdp

        # Feed-forward weights [n_l1, n_l2]
        rng = np.random.default_rng(seed=13)
        w_np = rng.uniform(l2.init_w_min, l2.init_w_max, (n_l1, n)).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # Pre-allocated input buffer
        self._x_buf = torch.zeros(n_l1, dtype=torch.float32)

        # snnTorch LIF with WTA inhibition
        self.lif = snn.Leaky(
            beta=beta,
            threshold=self.threshold,
            inhibition=True,
            reset_mechanism="zero",
        )
        self.mem = self.lif.init_leaky()

        # State tracking
        self.t: int = 0
        self.last_post_spike = np.full(n, -9999, dtype=np.int64)
        self.last_pre_spike = np.full(n_l1, -9999, dtype=np.int64)
        self.refractory = l2.refractory_samples

    def step(self, l1_spikes: np.ndarray) -> np.ndarray:
        """
        Advance one timestep.

        Parameters
        ----------
        l1_spikes : np.ndarray
            Boolean array ``[n_l1]`` — which L1 neurons fired.

        Returns
        -------
        np.ndarray
            Boolean array ``[n_l2]`` — which L2 neurons fired.
        """
        self.t += 1

        # Track pre-spike times from L1
        active = np.flatnonzero(l1_spikes)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

        # Build input tensor
        self._x_buf.zero_()
        if len(active) > 0:
            self._x_buf[active] = 1.0

        # Feed-forward current
        current = self._x_buf @ self.W  # [n_l2]

        # Lateral inhibition scaling: amplify WTA competition
        # The WTA in snnTorch subtracts the max membrane from non-winners;
        # wi_factor amplifies this effect by scaling the input spread
        if self.wi_factor > 1.0:
            mean_c = current.mean()
            current = mean_c + (current - mean_c) * self.wi_factor

        with torch.no_grad():
            spk, self.mem = self.lif(current.unsqueeze(0), self.mem.unsqueeze(0))
            spk = spk.squeeze(0)
            self.mem = self.mem.squeeze(0)

        spikes = spk.numpy().astype(bool)

        # Enforce refractory + STDP
        for w in np.flatnonzero(spikes):
            if (self.t - self.last_post_spike[w]) <= self.refractory:
                spikes[w] = False
                continue
            self.last_post_spike[w] = self.t
            if not self.freeze:
                self._stdp(int(w))

        return spikes

    def _stdp(self, winner: int) -> None:
        """Competitive asymmetric STDP on the winning neuron's column."""
        w = self.W[:, winner].numpy()

        # Global LTD
        w += self.ltd

        # LTP for recently active L1 neurons
        dt = self.t - self.last_pre_spike
        causal = (dt <= self.ltp_win) & (self.last_pre_spike > 0)
        w[causal] += self.ltp

        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = torch.from_numpy(w)
