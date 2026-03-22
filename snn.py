"""
snn.py — Spiking Neural Network: TemplateLayer (snnTorch backend).

ANNet-derived L1 pattern matching layer with competitive asymmetric STDP.
Uses snntorch.Leaky for LIF membrane dynamics.

Requires: torch, snntorch
"""

import numpy as np
import torch
import snntorch as snn


class TemplateLayer:
    """
    N LIF neurons that learn recurring spatio-temporal patterns in the
    encoded electrode signal via competitive asymmetric STDP.

    Each neuron becomes a template matcher for a particular waveform shape.
    Winner-take-all competition ensures distinct templates.

    Inputs per step:
        afferents : bool array [n_afferents]  — from SpikeEncoder
        dn_spike  : bool                      — from AttentionNeuron

    Output per step:
        spikes    : bool array [n_neurons]    — which L1 neurons fired
    """

    def __init__(self, cfg: dict, n_afferents: int):
        n = cfg["l1_n_neurons"]
        tm = cfg["l1_tm_samples"]
        beta = float(np.exp(-1.0 / tm))

        # Threshold (same formula as original ANNet derivation)
        k = 3
        overlap = cfg["enc_overlap"]
        window  = cfg["enc_window_depth"]
        step_sz = cfg["enc_step_size"]
        self.threshold = float(
            (cfg["l1_dn_weight"] + overlap * (window - k))
            * (1.0 - np.exp(-step_sz / tm))
            / (1.0 - np.exp(-1.0 / tm))
        )

        self.reset_v   = cfg["l1_reset_potential"]
        self.dn_weight = cfg["l1_dn_weight"]
        self.refractory = cfg["l1_refractory_samples"]
        self.freeze    = cfg["l1_freeze_stdp"]

        # STDP params
        self.ltp     = cfg["l1_stdp_ltp"]
        self.ltp_win = cfg["l1_stdp_ltp_window"]
        self.ltd     = cfg["l1_stdp_ltd"]
        w_lo, w_hi   = cfg["l1_w_bounds"]
        self.w_lo    = w_lo
        self.w_hi    = w_hi

        # Weights: [n_afferents × n_neurons]  uniform in [init_min, init_max]
        rng = np.random.default_rng(seed=7)
        w_np = rng.uniform(cfg["l1_init_w_min"], cfg["l1_init_w_max"],
                           (n_afferents, n)).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # snnTorch LIF (inhibition=True enables WTA)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=self.threshold,
            inhibition=True,
            reset_mechanism="zero",
        )
        self.mem = self.lif.init_leaky()

        # Neuron state
        self.n     = n
        self.n_aff = n_afferents
        self.t     = 0
        self.last_post_spike = np.full(n, -9999, dtype=np.int64)
        self.last_pre_spike  = np.full(n_afferents, -9999, dtype=np.int64)

    # ── public API ────────────────────────────────────────────────────────────
    def step(self, afferents: np.ndarray, dn_spike: bool) -> np.ndarray:
        """
        Advance one simulation step.
        Returns bool array of shape (n_neurons,) — which L1 neurons fired.
        """
        self.t += 1

        # Track pre-spike times
        active = np.flatnonzero(afferents)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

        # Compute input current: afferents @ W
        x = torch.from_numpy(afferents.astype(np.float32))
        current = x @ self.W  # [n_neurons]

        # DN excitation — add as current so LIF applies a single decay
        if dn_spike:
            current = current + self.dn_weight

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

    def get_weights(self) -> np.ndarray:
        """Return a copy of the weight matrix [n_afferents × n_neurons]."""
        return self.W.numpy().copy()

    # ── STDP ──────────────────────────────────────────────────────────────────
    def _stdp(self, winner: int) -> None:
        """
        Asymmetric Hebbian STDP on the winning neuron's weights.
          - LTD: all synapses depressed by ltd at post-spike
          - LTP: afferents active within ltp_window before post-spike potentiated
        """
        w = self.W[:, winner].numpy()

        # Global LTD on all synapses
        w += self.ltd

        # LTP for recently active afferents
        dt = self.t - self.last_pre_spike
        causal = (dt <= self.ltp_win) & (self.last_pre_spike > 0)
        w[causal] += self.ltp

        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = torch.from_numpy(w)

    # ── Encoding / decoding ───────────────────────────────────────────────────
    @staticmethod
    def encode(value: int, bit_width: int) -> np.ndarray:
        """
        Rate encoding: bit k of `value` drives input neuron k.
          bit = 1  →  Bernoulli spike with p = 0.9
          bit = 0  →  Bernoulli spike with p = 0.1
        Returns bool array of shape (bit_width,).
        """
        bits  = np.array([(value >> i) & 1 for i in range(bit_width)], dtype=np.float32)
        probs = bits * 0.8 + 0.1          # map:  0 → 0.1,  1 → 0.9
        return np.random.rand(bit_width) < probs

    @staticmethod
    def decode(spikes_out: np.ndarray) -> int:
        """
        Bit-threshold decoding: output neuron k spikes → bit k = 1 in result.
        Returns the reconstructed integer.
        """
        return int(sum(int(s) << i for i, s in enumerate(spikes_out)))
