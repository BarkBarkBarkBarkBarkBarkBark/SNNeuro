# AGENT-HINT: L1 template-matching layer with competitive STDP.
# PURPOSE: 110 LIF neurons (snnTorch Leaky + WTA) that learn spike waveform templates.
# INPUTS: afferents (encoder), dn_spike (attention), suppression (inhibition + noise_gate)
# STDP: Global LTD on every post-spike + causal LTP for recent pre-spikes.
# CONFIG: L1Config in config.py (n_neurons, dn_weight, stdp_ltp, stdp_ltd, etc.)
# PERF:  All tensors stay on CPU for single-sample throughput (avoids CUDA sync
#        overhead).  STDP is vectorized across all winners in one operation.
# SEE ALSO: attention.py (dn_spike input), inhibition.py (post-spike blanking),
#           noise_gate.py (noise suppression), dec_layer.py (DEC downstream)
"""
snn_agent.core.template — L1 template-matching layer with competitive STDP.

Uses snnTorch ``Leaky`` for LIF membrane dynamics.  Winner-take-all
competition ensures distinct spike templates.

**Vectorized STDP:** the per-winner Python for-loop has been replaced with
a fully batched operation over all winners simultaneously, eliminating the
#1 cause of slowdown when the attention neuron is very active.

All tensors remain on CPU to avoid per-sample CUDA synchronisation overhead
(benchmarked 3.5× slower on Jetson Orin for single-sample-at-a-time path).
"""

from __future__ import annotations

import numpy as np
import torch
import snntorch as snn

from snn_agent.config import Config

# Pin PyTorch to 1 thread — our matmuls are tiny (e.g. 2270×110);
# OpenMP thread-pool overhead far exceeds the compute savings.
torch.set_num_threads(1)

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

        # Weights [n_afferents × n_neurons] — CPU tensor
        rng = np.random.default_rng(seed=7)
        w_np = rng.uniform(l1.init_w_min, l1.init_w_max, (n_afferents, n)).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # Pre-allocate reusable tensors (avoids per-step allocation)
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
        self.last_current_magnitude: float = 0.0  # pre-suppression peak |I|

        # STDP timing as torch tensor for vectorized update
        self._last_pre_spike_t = torch.full(
            (n_afferents,), -9999, dtype=torch.int64
        )

        # Pre-allocated CPU output buffer
        self._spk_cpu = np.empty(n, dtype=bool)

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
        n_active = len(active)
        if n_active > 0:
            self.last_pre_spike[active] = self.t
            self._last_pre_spike_t[active] = self.t

        # Build input tensor (scatter active indices)
        self._x_buf.zero_()
        if n_active > 0:
            self._x_buf[active] = 1.0

        # Compute input current: afferents @ W
        current = self._x_buf @ self.W  # [n_neurons]

        if dn_spike:
            current = current + self.dn_weight

        # Expose pre-suppression peak for inhibitor bypass decision
        self.last_current_magnitude = float(current.max().item())

        # Apply suppression from noise gate / inhibitor
        if suppression < 1.0:
            current = current * suppression

        # LIF forward pass
        with torch.no_grad():
            spk, self.mem = self.lif(current.unsqueeze(0), self.mem.unsqueeze(0))
            spk = spk.squeeze(0)
            self.mem = self.mem.squeeze(0)

        # Convert spikes to bool numpy
        np.greater(spk.numpy(), 0.5, out=self._spk_cpu)

        # Refractory enforcement (CPU — small array, branchy logic)
        winners_raw = np.flatnonzero(self._spk_cpu)
        if len(winners_raw) > 0:
            dt = self.t - self.last_post_spike[winners_raw]
            refractory_mask = dt <= self.refractory
            # Kill refractory spikes
            if refractory_mask.any():
                self._spk_cpu[winners_raw[refractory_mask]] = False

            # Valid winners (non-refractory)
            valid_winners = winners_raw[~refractory_mask]
            if len(valid_winners) > 0:
                self.last_post_spike[valid_winners] = self.t
                if not self.freeze:
                    self._stdp_vectorized(valid_winners)

        return self._spk_cpu

    # ── Vectorized STDP (batched, no Python for-loop) ───────────────
    def _stdp_vectorized(self, winners: np.ndarray) -> None:
        """
        Asymmetric Hebbian STDP on all winning neurons simultaneously.

        Instead of looping over each winner and copying columns to/from
        numpy, this operates directly on torch tensors in a single batched
        operation.
        """
        n_winners = len(winners)
        if n_winners == 0:
            return

        win_idx = torch.from_numpy(winners.astype(np.int64))

        # Extract winning columns: [n_afferents, n_winners]
        w_cols = self.W[:, win_idx]

        # Global LTD: all afferent weights decrease
        w_cols = w_cols + self.ltd

        # LTP for recently active afferents (causal STDP)
        dt = self.t - self._last_pre_spike_t  # [n_afferents]
        causal = (dt <= self.ltp_win) & (self._last_pre_spike_t > 0)
        # Broadcast causal mask across all winners: [n_afferents, 1]
        ltp_delta = causal.unsqueeze(1).float() * self.ltp
        w_cols = w_cols + ltp_delta

        # Clamp to bounds
        w_cols.clamp_(self.w_lo, self.w_hi)

        # Write back (in-place scatter)
        self.W[:, win_idx] = w_cols
