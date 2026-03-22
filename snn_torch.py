"""
snn_torch.py — snnTorch-backed template layer (drop-in for TemplateLayer).

Requires: pip install snntorch torch

Uses snntorch.Leaky for LIF dynamics with WTA inhibition.
STDP remains custom (snnTorch does not provide it).
"""

import numpy as np
import torch
import snntorch as snn


class TorchTemplateLayer:
    """Drop-in replacement for TemplateLayer using snnTorch + PyTorch."""

    def __init__(self, cfg: dict, n_afferents: int):
        n = cfg["l1_n_neurons"]
        tm = cfg["l1_tm_samples"]
        beta = float(np.exp(-1.0 / tm))

        # Threshold (same formula as NumPy version)
        k = 3
        overlap = cfg["enc_overlap"]
        window = cfg["enc_window_depth"]
        step_sz = cfg["enc_step_size"]
        self.threshold = float(
            (cfg["l1_dn_weight"] + overlap * (window - k))
            * (1.0 - np.exp(-step_sz / tm))
            / (1.0 - np.exp(-1.0 / tm))
        )

        self.dn_weight = cfg["l1_dn_weight"]
        self.refractory = cfg["l1_refractory_samples"]
        self.freeze = cfg["l1_freeze_stdp"]

        # STDP params
        self.ltp = cfg["l1_stdp_ltp"]
        self.ltp_win = cfg["l1_stdp_ltp_window"]
        self.ltd = cfg["l1_stdp_ltd"]
        self.w_lo, self.w_hi = cfg["l1_w_bounds"]

        # Weights [n_afferents × n_neurons]
        rng = np.random.default_rng(seed=7)
        w_np = rng.uniform(cfg["l1_init_w_min"], cfg["l1_init_w_max"],
                           (n_afferents, n)).astype(np.float32)
        self.W = torch.from_numpy(w_np)

        # snnTorch LIF with WTA inhibition
        self.lif = snn.Leaky(
            beta=beta,
            threshold=self.threshold,
            inhibition=True,
            reset_mechanism="zero",
        )
        self.mem = self.lif.init_leaky()

        # State for STDP / refractory tracking
        self.n = n
        self.n_aff = n_afferents
        self.t = 0
        self.last_post_spike = np.full(n, -9999, dtype=np.int64)
        self.last_pre_spike = np.full(n_afferents, -9999, dtype=np.int64)

    def step(self, afferents: np.ndarray, dn_spike: bool) -> np.ndarray:
        self.t += 1

        x = torch.from_numpy(afferents.astype(np.float32))
        current = x @ self.W  # [n_neurons]
        if dn_spike:
            current += self.dn_weight

        with torch.no_grad():
            spk, self.mem = self.lif(current.unsqueeze(0), self.mem.unsqueeze(0))
            spk = spk.squeeze(0)
            self.mem = self.mem.squeeze(0)

        spikes = spk.numpy().astype(bool)

        # Track pre-spike times
        active = np.flatnonzero(afferents)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

        # Enforce refractory + STDP on winners
        for w in np.flatnonzero(spikes):
            if (self.t - self.last_post_spike[w]) <= self.refractory:
                spikes[w] = False
                continue
            self.last_post_spike[w] = self.t
            if not self.freeze:
                self._stdp(w)

        return spikes

    def get_weights(self) -> np.ndarray:
        return self.W.numpy().copy()

    def _stdp(self, winner: int) -> None:
        w = self.W[:, winner].numpy()
        w += self.ltd
        dt = self.t - self.last_pre_spike
        causal = (dt <= self.ltp_win) & (self.last_pre_spike > 0)
        w[causal] += self.ltp
        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = torch.from_numpy(w)
