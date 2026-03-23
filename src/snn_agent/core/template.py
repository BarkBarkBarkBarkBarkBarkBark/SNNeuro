# AGENT-HINT: L1 template-matching layer with competitive STDP.
# PURPOSE: 110 LIF neurons (snnTorch Leaky + WTA) that learn spike waveform templates.
# INPUTS: afferents (encoder), dn_spike (attention), suppression (inhibition + noise_gate)
# STDP: Global LTD on every post-spike + causal LTP for recent pre-spikes.
# CONFIG: L1Config in config.py (n_neurons, dn_weight, stdp_ltp, stdp_ltd, etc.)
# SEE ALSO: attention.py (dn_spike input), inhibition.py (post-spike blanking),
#           noise_gate.py (noise suppression), output_layer.py (optional L2 downstream)
"""
snn_agent.core.template — L1 template-matching layer with competitive STDP.

Uses snnTorch ``Leaky`` for LIF membrane dynamics.  Winner-take-all
competition ensures distinct spike templates.

NumPy ↔ Torch conversions are minimised: the weight matrix and membrane
state are kept as Torch tensors throughout; only the final spike output
is converted to a NumPy bool array.
"""

from __future__ import annotations

import numpy as np
import torch
import snntorch as snn

from snn_agent.config import Config

__all__ = ["TemplateLayer"]


class TemplateLayer:
    """
    N LIF neurons that learn recurring spatio-temporal patterns via
    competitive asymmetric STDP.

    Each neuron becomes a template matcher for a particular waveform shape.

    Inputs per step:
        ``afferents`` — bool array ``[n_afferents]`` from SpikeEncoder
        ``dn_spike``  — bool from AttentionNeuron

    Output per step:
        ``spikes``    — bool array ``[n_neurons]`` indicating which fired
    """

    def __init__(self, cfg: Config, n_afferents: int) -> None:
        l1 = cfg.l1
        enc = cfg.encoder
        n = l1.n_neurons
        tm = l1.tm_samples
        beta = float(np.exp(-1.0 / tm))

        # Threshold (ANNet derivation)
        k = 3
        self.threshold = float(
            (l1.dn_weight + enc.overlap * (enc.window_depth - k))
            * (1.0 - np.exp(-enc.step_size / tm))
            / (1.0 - np.exp(-1.0 / tm))
        )

        self.dn_weight = l1.dn_weight
        self.refractory = l1.refractory_samples
        self.freeze = l1.freeze_stdp

        # STDP params
        self.ltp = l1.stdp_ltp
        self.ltp_win = l1.stdp_ltp_window
        self.ltd = l1.stdp_ltd
        self.w_lo = l1.w_lo
        self.w_hi = l1.w_hi

        # Weights [n_afferents × n_neurons] — kept as torch tensor
        rng = np.random.default_rng(seed=7)
        w_np = rng.uniform(l1.init_w_min, l1.init_w_max, (n_afferents, n)).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # Pre-allocate a reusable float32 input tensor
        self._x_buf = torch.zeros(n_afferents, dtype=torch.float32)

        # snnTorch LIF (inhibition=True → WTA)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=self.threshold,
            inhibition=True,
            reset_mechanism="zero",
        )
        self.mem = self.lif.init_leaky()

        # Neuron state
        self.n = n
        self.n_aff = n_afferents
        self.t: int = 0
        self.last_post_spike = np.full(n, -9999, dtype=np.int64)
        self.last_pre_spike = np.full(n_afferents, -9999, dtype=np.int64)

    # ── public API ────────────────────────────────────────────────────
    def step(
        self,
        afferents: np.ndarray,
        dn_spike: bool,
        suppression: float = 1.0,
    ) -> np.ndarray:
        """
        Advance one simulation step.

        Parameters
        ----------
        afferents : np.ndarray
            Bool array ``[n_afferents]`` from SpikeEncoder.
        dn_spike : bool
            Whether the attention neuron fired this step.
        suppression : float
            Multiplicative suppression factor from noise gate / inhibitor.
            1.0 = no suppression, 0.0 = fully suppressed.

        Returns bool array ``(n_neurons,)`` — which L1 neurons fired.
        """
        self.t += 1

        # Track pre-spike times
        active = np.flatnonzero(afferents)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

        # Copy afferents into pre-allocated tensor (avoids per-step allocation)
        self._x_buf.zero_()
        if len(active) > 0:
            self._x_buf[active] = 1.0

        # Compute input current: afferents @ W
        current = self._x_buf @ self.W  # [n_neurons]

        if dn_spike:
            current = current + self.dn_weight

        # Apply suppression from noise gate / inhibitor
        if suppression < 1.0:
            current = current * suppression

        with torch.no_grad():
            spk, self.mem = self.lif(current.unsqueeze(0), self.mem.unsqueeze(0))
            spk = spk.squeeze(0)
            self.mem = self.mem.squeeze(0)

        spikes = spk.numpy().astype(bool)

        # Enforce refractory + STDP on winners
        for w in np.flatnonzero(spikes):
            if (self.t - self.last_post_spike[w]) <= self.refractory:
                spikes[w] = False
                continue
            self.last_post_spike[w] = self.t
            if not self.freeze:
                self._stdp(int(w))

        return spikes

    # ── STDP ──────────────────────────────────────────────────────────
    def _stdp(self, winner: int) -> None:
        """Asymmetric Hebbian STDP on the winning neuron's column."""
        w = self.W[:, winner].numpy()

        # Global LTD
        w += self.ltd

        # LTP for recently active afferents
        dt = self.t - self.last_pre_spike
        causal = (dt <= self.ltp_win) & (self.last_pre_spike > 0)
        w[causal] += self.ltp

        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = torch.from_numpy(w)
