"""
snn_agent.core.batched — CUDA-batched neural layers for multi-channel processing.

BatchedTemplateLayer and BatchedDECLayer process C channels in a single
batched forward pass.  Weights, membranes, and STDP state are kept on
the target device (CUDA when available) to avoid CPU↔GPU transfers.

Single-channel (C=1) is fully supported — the batch dimension is just 1.
"""

from __future__ import annotations

import numpy as np
import torch

from snn_agent.config import Config
from snn_agent.core._numba_kernels import (
    attention_block,
    attention_block_parallel,
    template_lif_wta_block,
    template_lif_wta_block_parallel,
)

__all__ = ["BatchedAttentionNeuron", "BatchedTemplateLayer", "BatchedDECLayer", "ConvergenceLayer"]


# ---------------------------------------------------------------------------
#  BatchedAttentionNeuron  — vectorised across C channels
# ---------------------------------------------------------------------------
class BatchedAttentionNeuron:
    """
    C-channel vectorised attention (DN) neuron.

    Replaces C independent ``AttentionNeuron`` instances with a single
    ``[C, max_aff]`` NumPy pass — no Python loop over channels.

    State arrays are padded to *max_aff*; positions beyond each channel's
    real ``n_afferents`` are never activated by the encoder so they stay
    permanently zero.
    """

    def __init__(self, cfg: Config, n_afferents_per_ch: list[int]) -> None:
        dn = cfg.dn
        enc = cfg.encoder
        C = len(n_afferents_per_ch)
        tm = dn.tm_samples
        max_aff = max(n_afferents_per_ch)

        self.C = C
        self.max_aff = max_aff
        self.threshold = float(
            dn.threshold_factor * enc.overlap * enc.window_depth
            / (1.0 - np.exp(-1.0 / tm))
        )
        self.reset_potential = float(
            dn.reset_potential_factor * (np.exp(1.0 / tm) - 1.0) * self.threshold
        )
        self.f_d = float(dn.depression_frac)
        self.decay = float(np.exp(-1.0 / tm))

        # Synaptic depression LUT (index = Δt in samples, clipped to max_dt)
        self._max_dt = max(1, int(dn.depression_tau * 20))
        self._exp_td = np.exp(
            -np.arange(self._max_dt + 1, dtype=np.float32) / dn.depression_tau
        )

        # State [C, max_aff]
        self.p_rel = np.ones((C, max_aff), dtype=np.float32)
        self.last_pre = np.full((C, max_aff), -9999, dtype=np.int64)
        self.v = np.zeros(C, dtype=np.float64)
        self.t: int = 0

        # Pre-allocated scratch (avoid per-step allocations)
        self._dt_buf      = np.zeros((C, max_aff), dtype=np.int64)
        self._p_rec       = np.zeros((C, max_aff), dtype=np.float32)
        self._contrib_np  = np.zeros(C, dtype=np.float32)  # fused dot-product result

        # Block API pre-allocated buffers (64-step default; grown on demand)
        _BN = 64
        self._block_N = _BN
        self._dn_block = np.zeros((_BN, C), dtype=bool)

    def step_batch(self, aff_bool: np.ndarray) -> np.ndarray:
        """
        One timestep for all C channels.

        Parameters
        ----------
        aff_bool : [C, max_aff] bool — active afferents (from pre-allocated buffer).

        Returns
        -------
        fired : [C] bool
        """
        self.t += 1
        t = self.t

        # Delta-time for all (ch, afferent) pairs — reuse pre-allocated buffer
        np.subtract(t, self.last_pre, out=self._dt_buf)
        np.clip(self._dt_buf, 0, self._max_dt, out=self._dt_buf)

        # Synaptic recovery: p_rec = 1 − (1 − p_rel) · exp(−Δt/τ)
        p_rec = self._p_rec
        np.subtract(1.0, self.p_rel, out=p_rec)
        p_rec *= self._exp_td[self._dt_buf]  # LUT lookup, in-place multiply
        np.subtract(1.0, p_rec, out=p_rec)   # p_rec = recovered p_rel

        # Fused row dot-product: sum of p_rec where aff_bool is True, per channel
        # einsum 'ca,ca->c' avoids the [C, max_aff] intermediate that (*).sum() creates
        np.einsum('ca,ca->c', p_rec, aff_bool, out=self._contrib_np)
        contrib_sum = self._contrib_np

        # Update p_rel for fired positions only (in-place masked write — no np.where array)
        np.multiply(p_rec, (1.0 - self.f_d), where=aff_bool, out=self.p_rel)

        # Update last_pre for fired positions (direct bool-index write — no np.where array)
        self.last_pre[aff_bool] = t

        # LIF membrane update
        self.v *= self.decay
        self.v += contrib_sum

        # Fire + soft-reset
        fired = self.v >= self.threshold
        self.v[fired] = self.reset_potential
        return fired  # [C] bool

    def step_block(
        self,
        aff_bool_block: np.ndarray,
        *,
        parallel: bool = False,
    ) -> np.ndarray:
        """
        Process N timesteps for all C channels using a Numba JIT kernel.

        Parameters
        ----------
        aff_bool_block : [N, C, max_aff] bool  — pre-allocated block buffer.
        parallel : bool
            Use the ``prange(C)`` parallel Numba variant.  Safe on Jetson Orin
            when C ≥ 8; may incur thread-sync overhead for smaller C.

        Returns
        -------
        dn_block : [N, C] bool  — view into pre-allocated ``_dn_block`` buffer.
        """
        N = aff_bool_block.shape[0]
        if N > self._block_N:
            self._block_N = max(N, 64)
            self._dn_block = np.zeros((self._block_N, self.C), dtype=bool)
        dn_out = self._dn_block[:N]

        kern = attention_block_parallel if parallel else attention_block
        kern(
            aff_bool_block,
            self._exp_td,
            self.p_rel,
            self.last_pre,
            self.v,
            dn_out,
            np.int64(self.t),
            np.float64(self.decay),
            np.float64(self.threshold),
            np.float64(self.reset_potential),
            np.float32(self.f_d),
            np.int64(self._max_dt),
        )
        self.t += N
        return dn_out  # [N, C] bool — view into _dn_block


# ---------------------------------------------------------------------------
#  BatchedTemplateLayer
# ---------------------------------------------------------------------------
class BatchedTemplateLayer:
    """
    C-channel batched L1 template layer with WTA + competitive STDP.

    Weights ``W`` have shape ``[C, max_aff, n_neurons]`` on *device*.
    Each channel has independent membrane state, STDP tracking, and
    refractory counters.
    """

    def __init__(
        self,
        cfg: Config,
        n_afferents_per_ch: list[int],
        device: torch.device,
    ) -> None:
        l1 = cfg.l1
        enc = cfg.encoder
        C = len(n_afferents_per_ch)
        n = l1.n_neurons
        tm = l1.tm_samples
        beta = float(np.exp(-1.0 / tm))

        self.C = C
        self.n = n
        self.device = device
        self.n_afferents_per_ch = list(n_afferents_per_ch)
        self.max_aff = max(n_afferents_per_ch)

        # Threshold (ANNet derivation — same formula as TemplateLayer)
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

        # Weights [C, max_aff, n_neurons] — NumPy for zero-overhead CPU matmul
        rng = np.random.default_rng(seed=7)
        self.W_np = np.zeros((C, self.max_aff, n), dtype=np.float32)
        for ch in range(C):
            naff = n_afferents_per_ch[ch]
            self.W_np[ch, :naff, :] = rng.uniform(l1.init_w_min, l1.init_w_max, (naff, n)).astype(np.float32)
        # Keep torch view for any caller expecting .W attribute (e.g. app.py membrane read)
        self.W = torch.from_numpy(self.W_np)

        # Inline LIF parameters
        self.beta = beta
        self.mem_np = np.zeros((C, n), dtype=np.float32)

        # Per-channel STDP state — NumPy
        self._last_pre_np = np.full((C, self.max_aff), -9999, dtype=np.int64)
        self._last_post_np = np.full((C, n), -9999, dtype=np.int64)
        self.t: int = 0

        # Current magnitude for inhibitor bypass (CPU float array)
        self._last_current_mag_np = np.zeros(C, dtype=np.float32)
        # Torch tensor view for multichannel.py compatibility
        self.last_current_magnitude = torch.from_numpy(self._last_current_mag_np)

        # Pre-allocated output spike buffer — shared memory between numpy and torch
        # torch.from_numpy creates a zero-copy view: modifying _spk_np updates _spk_t automatically
        self._spk_np = np.zeros((C, n), dtype=bool)
        self._spk_t = torch.from_numpy(self._spk_np)  # shared memory, no copy_ needed
        # Pre-allocated current buffer (reused in step_sparse as scratch; also used by Numba block)
        self._cur_np = np.zeros((C, n), dtype=np.float32)
        # Membrane tensor (lazy-built on request for app.py viz)
        self._mem_t: torch.Tensor | None = None

        # Block API pre-allocated buffers (64-step default; grown on demand)
        _BN = 64
        self._block_N = _BN
        self._spk_block = np.zeros((_BN, C, n), dtype=bool)

    def step(
        self,
        afferents_batch: torch.Tensor,
        dn_spikes: torch.Tensor,
        suppressions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward one timestep for all C channels.

        Parameters
        ----------
        afferents_batch : Tensor [C, max_aff]  float32 (any device)
        dn_spikes       : Tensor [C]           bool
        suppressions    : Tensor [C]            float32

        Returns
        -------
        spikes : Tensor [C, n_neurons] bool (CPU)
        """
        self.t += 1

        # Pull to NumPy (CPU path — avoids CUDA kernel dispatch overhead for tiny tensors)
        aff_np = afferents_batch.detach().cpu().numpy()  # [C, max_aff]
        dn_np = dn_spikes.detach().cpu().numpy().astype(np.float32)   # [C]
        sup_np = suppressions.detach().cpu().numpy().astype(np.float32)  # [C]

        # Track pre-spike times
        active_mask = aff_np > 0.5  # [C, max_aff]
        self._last_pre_np[active_mask] = self.t

        # Input current: batched matmul [C, max_aff] × [C, max_aff, n] → [C, n]
        # Use einsum with optimal contraction ordering for BLAS dispatch
        current = np.einsum('ca,can->cn', aff_np, self.W_np, optimize=False)  # [C, n]

        # DN excitatory boost
        current += self.dn_weight * dn_np[:, None]

        # Pre-suppression magnitude for inhibitor bypass
        self._last_current_mag_np[:] = current.max(axis=1)

        # Apply suppression
        current *= sup_np[:, None]

        # LIF membrane update
        self.mem_np *= self.beta
        self.mem_np += current

        # WTA: per-row argmax → single winner per channel
        winners = self.mem_np.argmax(axis=1)  # [C]
        any_above = self.mem_np[np.arange(self.C), winners] >= self.threshold  # [C]
        spk = self._spk_np
        spk[:] = False
        if any_above.any():
            firing_ch = np.where(any_above)[0]
            spk[firing_ch, winners[firing_ch]] = True
        self.mem_np[spk] = 0.0

        # Refractory + STDP (vectorized where possible)
        if spk.any():
            spk_chs, spk_ns = np.where(spk)
            refrac_ok = (self.t - self._last_post_np[spk_chs, spk_ns]) > self.refractory
            # Cancel refrac violations
            cancel = ~refrac_ok
            if cancel.any():
                spk[spk_chs[cancel], spk_ns[cancel]] = False
            # Update last_post and run STDP for kept spikes
            kept = refrac_ok
            if kept.any():
                kch, kn = spk_chs[kept], spk_ns[kept]
                self._last_post_np[kch, kn] = self.t
                if not self.freeze:
                    for ch, w in zip(kch.tolist(), kn.tolist()):
                        self._stdp(ch, w)

        self._spk_t.copy_(torch.from_numpy(spk))
        return self._spk_t

    def step_sparse(
        self,
        aff_bool: np.ndarray,
        aff_f32: np.ndarray,
        dn_np: np.ndarray,
        sup_np: np.ndarray,
    ) -> np.ndarray:
        """
        Sparse-aware forward pass.

        Uses ``aff_bool`` ([C, max_aff] bool) for two things:
        · Vectorized ``_last_pre_np[aff_bool] = t`` STDP timing (single C call,
          no Python for-loop over channels).
        · Dense BLAS matmul via ``aff_f32`` ([C, max_aff] float32).

        Parameters
        ----------
        aff_bool : [C, max_aff] bool  — afferent buffer (pre-alloc, shared memory)
        aff_f32  : [C, max_aff] float32  — float32 companion (same data, pre-cast)
        dn_np    : [C] float32 — DN spike flags
        sup_np   : [C] float32 — suppression factors

        Returns
        -------
        _spk_np : [C, n_neurons] bool (shared memory with _spk_t)
        """
        self.t += 1

        # Vectorized pre-spike timing update for STDP — single bool-index write,
        # replaces the old Python for-loop + per-channel fancy-index writes.
        self._last_pre_np[aff_bool] = self.t

        # Single BLAS call: [C, max_aff] @ [C, max_aff, n] → [C, n]
        # ~10× faster than 16 Python .sum(axis=0) calls for C=16
        np.einsum('ca,can->cn', aff_f32, self.W_np, out=self._cur_np, optimize=False)
        current = self._cur_np  # alias for readability below

        # DN excitatory boost
        current += self.dn_weight * dn_np[:, np.newaxis]

        # Pre-suppression magnitude for inhibitor bypass
        self._last_current_mag_np[:] = current.max(axis=1)

        # Apply suppression
        current *= sup_np[:, np.newaxis]

        # LIF membrane update
        self.mem_np *= self.beta
        self.mem_np += current

        # WTA: argmax winner per channel
        winners = self.mem_np.argmax(axis=1)  # [C]
        any_above = self.mem_np[np.arange(self.C), winners] >= self.threshold
        spk = self._spk_np
        spk[:] = False
        if any_above.any():
            firing_ch = np.where(any_above)[0]
            spk[firing_ch, winners[firing_ch]] = True
        self.mem_np[spk] = 0.0

        # Refractory + STDP
        if spk.any():
            spk_chs, spk_ns = np.where(spk)
            refrac_ok = (self.t - self._last_post_np[spk_chs, spk_ns]) > self.refractory
            if not refrac_ok.all():
                spk[spk_chs[~refrac_ok], spk_ns[~refrac_ok]] = False
            kept = refrac_ok
            if kept.any():
                kch, kn = spk_chs[kept], spk_ns[kept]
                self._last_post_np[kch, kn] = self.t
                if not self.freeze:
                    for ch, w in zip(kch.tolist(), kn.tolist()):
                        self._stdp(ch, w)

        # _spk_t is a shared-memory view of _spk_np — no copy needed
        return self._spk_np

    def step_sparse_block(
        self,
        aff_bool_block: np.ndarray,   # [N, C, max_aff] bool
        aff_f32_block: np.ndarray,    # [N, C, max_aff] float32
        dn_block: np.ndarray,         # [N, C] float32
        sup_block: np.ndarray,        # [N, C] float32
        *,
        parallel: bool = False,
    ) -> np.ndarray:
        """
        Block forward pass via Numba JIT kernel.

        Processes N timesteps in a single compiled call, eliminating per-step
        Python dispatch overhead for einsum, LIF, WTA, and STDP.

        Parameters
        ----------
        aff_bool_block : [N, C, max_aff] bool
        aff_f32_block  : [N, C, max_aff] float32
        dn_block       : [N, C] float32  — DN spike flags as float
        sup_block      : [N, C] float32  — suppression factors
        parallel       : bool  — use prange(C) Numba variant

        Returns
        -------
        spk_block : [N, C, n_neurons] bool — view into pre-allocated ``_spk_block``.
            ``spk_block[i]`` is step i's spike output.  Valid until the next
            call to ``step_sparse_block`` (pre-alloc is reused).
        """
        N = aff_bool_block.shape[0]
        if N > self._block_N:
            self._block_N = max(N, 64)
            self._spk_block = np.zeros((self._block_N, self.C, self.n), dtype=bool)
        spk_out = self._spk_block[:N]

        kern = template_lif_wta_block_parallel if parallel else template_lif_wta_block
        kern(
            aff_bool_block,
            aff_f32_block,
            self.W_np,
            self.mem_np,
            self._last_pre_np,
            self._last_post_np,
            spk_out,
            self._cur_np,         # scratch — safe to reuse; kernel clears it per (ti,ch)
            self._last_current_mag_np,
            dn_block,
            sup_block,
            np.int64(self.t),
            np.float32(self.beta),
            np.float32(self.threshold),
            np.int64(self.refractory),
            np.float32(self.dn_weight),
            bool(self.freeze),
            np.float32(self.ltp),
            np.float32(self.ltd),
            np.int64(self.ltp_win),
            np.float32(self.w_lo),
            np.float32(self.w_hi),
        )
        self.t += N
        # Keep _spk_np in sync with last step for single-step API compatibility
        self._spk_np[:] = spk_out[-1]
        return spk_out

    def _stdp(self, ch: int, winner: int) -> None:
        """NumPy STDP for one channel's winning neuron."""
        w = self.W_np[ch, :, winner]
        w += self.ltd
        dt = self.t - self._last_pre_np[ch]
        causal = (dt <= self.ltp_win) & (dt > 0)
        w[causal] += self.ltp
        np.clip(w, self.w_lo, self.w_hi, out=w)

    @property
    def mem(self) -> torch.Tensor:
        """Membrane potential as Tensor (lazy, for viz only)."""
        if self._mem_t is None or self._mem_t.shape != (self.C, self.n):
            self._mem_t = torch.from_numpy(self.mem_np)
        return self._mem_t


# ---------------------------------------------------------------------------
#  BatchedDECLayer
# ---------------------------------------------------------------------------
class BatchedDECLayer:
    """
    C-channel batched DEC spiking decoder (16 neurons per channel).

    Neuron 0 = DN-gated any-fire detector (fixed weights).
    Neurons 1-15 = competitive STDP unit learners (WTA).
    DN gating is per-channel.
    """

    N_DEC = 16

    def __init__(
        self,
        cfg: Config,
        n_l1: int,
        C: int,
        device: torch.device,
    ) -> None:
        dc = cfg.dec
        self.C = C
        self.n_l1 = n_l1
        self.device = device
        n_unit = self.N_DEC - 1  # 15 learned neurons

        # Delay expansion parameters
        self.use_delays = dc.use_delays
        self.n_taps = dc.n_delay_taps if dc.use_delays else 1
        self.n_input = n_l1 * self.n_taps
        # NOTE: _delay_buf_np (below) is the live ring buffer used in step_sparse.
        # The old torch.zeros version was never read in the hot path — removed.

        # Inline LIF parameters (replaces snnTorch Leaky — avoids WTA overhead)
        tm = dc.tm_samples
        self.beta = float(np.exp(-1.0 / tm))

        # Thresholds
        self.any_fire_threshold = dc.any_fire_threshold
        self.unit_threshold = float(dc.unit_threshold_factor * n_l1)

        # Neuron 0: any-fire (fixed weights = 1, CPU)
        self.mem_any_np = np.zeros(C, dtype=np.float32)

        # Neurons 1-15: WTA learned (CPU NumPy — avoids CUDA dispatch overhead)
        self.mem_unit_np = np.zeros((C, n_unit), dtype=np.float32)

        # Pre-allocated pow2 bitmask for hex output (NumPy, CPU)
        self._pow2_np = (2 ** np.arange(self.N_DEC, dtype=np.int64))

        # Learnable weights [C, n_input, n_unit] — NumPy
        rng = np.random.default_rng(seed=17)
        self.W_np = np.zeros((C, self.n_input, n_unit), dtype=np.float32)
        for ch in range(C):
            self.W_np[ch] = rng.uniform(
                dc.init_w_min, dc.init_w_max, (self.n_input, n_unit)
            ).astype(np.float32)
        # Keep torch view for any callers expecting .W attribute
        self.W = torch.from_numpy(self.W_np)

        # Lateral inhibition
        self.wi_factor = dc.wi_factor

        # STDP
        self.ltp = dc.stdp_ltp
        self.ltp_win = dc.stdp_ltp_window
        self.ltd = dc.stdp_ltd
        self.w_lo = dc.w_lo
        self.w_hi = dc.w_hi
        self.freeze = dc.freeze_stdp

        # State (NumPy)
        self.n_unit = n_unit
        self.t: int = 0
        self.refractory = dc.refractory_samples
        self._last_pre_np = np.full((C, self.n_input), -9999, dtype=np.int64)
        self._last_post_np = np.full((C, n_unit), -9999, dtype=np.int64)

        # DN integration window (NumPy)
        _efs = cfg.effective_fs()
        self._dn_window_samples = max(1, int(dc.dn_window_ms * 1e-3 * _efs))
        self._dn_countdown_np = np.zeros(C, dtype=np.int32)

        # Delay buffer (NumPy)
        self._delay_buf_np = np.zeros((C, self.n_taps, n_l1), dtype=bool)

        # Per-channel hex output cache
        self._last_hex = np.zeros(C, dtype=np.uint16)

        # Pre-allocated buffers — shared memory between numpy and torch
        self._out_np    = np.zeros((C, self.N_DEC), dtype=bool)
        self._spk_u_np  = np.zeros((C, n_unit),     dtype=bool)
        self._cur_unit  = np.zeros((C, n_unit),      dtype=np.float32)
        # Zero-copy torch view: writing _out_np is immediately visible through _out_t
        self._out_t = torch.from_numpy(self._out_np)
        # Scratch buffers for step_sparse — pre-allocated to avoid per-step heap pressure
        self._pre_mask_np    = np.zeros((C, self.n_input), dtype=bool)
        self._active_mask_np = np.zeros(C, dtype=bool)
        self._spk0_buf       = np.zeros(C, dtype=bool)
        self._any_bool_buf   = np.zeros(C, dtype=bool)
        self._any_f32_buf    = np.zeros(C, dtype=np.float32)

        # Block API: output buffer [block_N, C, N_DEC] — grown on demand
        _BN = 64
        self._block_N = _BN
        self._out_block = np.zeros((_BN, C, self.N_DEC), dtype=bool)
        self._hex_block = np.zeros((_BN, C), dtype=np.uint16)

    @property
    def hex_output(self) -> np.ndarray:
        """Last output as uint16 bitmask per channel [C]."""
        return self._last_hex

    def step(
        self,
        l1_spikes_batch: torch.Tensor,
        dn_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward one timestep for all C channels.

        Parameters
        ----------
        l1_spikes_batch : Tensor [C, n_l1] bool
        dn_spikes       : Tensor [C] bool

        Returns
        -------
        out : Tensor [C, 16] bool (CPU)
        """
        self.t += 1
        out_np = self._out_np
        out_np[:] = False

        # Pull l1 spikes to numpy (already on CPU if device=cpu, or 1 sync)
        l1b_np = l1_spikes_batch[:, : self.n_l1].detach().cpu().numpy()

        # Delay expansion (NumPy ring buffer)
        if self.use_delays:
            self._delay_buf_np[:, 1:] = self._delay_buf_np[:, :-1]
            self._delay_buf_np[:, 0] = l1b_np
            stacked = self._delay_buf_np[:, ::-1, :].reshape(self.C, self.n_input)
        else:
            stacked = l1b_np  # [C, n_input]

        # DN countdown (NumPy)
        dn_np = dn_spikes.detach().cpu().numpy()
        wn = self._dn_window_samples
        self._dn_countdown_np = np.where(dn_np, wn, np.maximum(self._dn_countdown_np - 1, 0))
        active_mask = self._dn_countdown_np > 0  # [C] bool

        if not active_mask.any():
            self._last_hex[:] = 0
            self._out_t.zero_()
            return self._out_t

        # Pre-spike time tracking
        pre_mask = active_mask[:, None] & stacked  # [C, n_input]
        self._last_pre_np[pre_mask] = self.t
        x_all = pre_mask.astype(np.float32)  # [C, n_input]

        # Neuron 0: any-fire — LIF (sum all inputs)
        any_current = x_all.sum(axis=1)  # [C]  (all w=1)
        self.mem_any_np *= self.beta
        self.mem_any_np += any_current
        spk0 = self.mem_any_np >= self.any_fire_threshold  # [C] bool
        self.mem_any_np[spk0] = 0.0
        out_np[:, 0] = spk0

        # Neurons 1-15: LIF + WTA (batched matmul)
        # x_all [C, n_input] @ W_np [C, n_input, n_unit] → [C, n_unit]
        current = np.matmul(x_all[:, np.newaxis, :], self.W_np)[:, 0, :]  # [C, n_unit]

        if self.wi_factor > 1.0:
            mc = current.mean(axis=1, keepdims=True)
            current = mc + (current - mc) * self.wi_factor

        self.mem_unit_np *= self.beta
        self.mem_unit_np += current

        # WTA: argmax per row
        winners_u = self.mem_unit_np.argmax(axis=1)  # [C]
        any_above_u = self.mem_unit_np[np.arange(self.C), winners_u] >= self.unit_threshold  # [C]
        spk_u = self._spk_u_np
        spk_u[:] = False
        if any_above_u.any():
            firing_ch = np.where(any_above_u)[0]
            spk_u[firing_ch, winners_u[firing_ch]] = True
        self.mem_unit_np[spk_u] = 0.0

        # Refractory + STDP (only firing, active channels)
        for ch in np.where(active_mask & spk_u.any(axis=1))[0]:
            w_idx = int(winners_u[ch])
            if (self.t - self._last_post_np[ch, w_idx]) <= self.refractory:
                spk_u[ch, w_idx] = False
            else:
                self._last_post_np[ch, w_idx] = self.t
                if not self.freeze:
                    self._stdp(ch, w_idx)

        out_np[active_mask, 1:] = spk_u[active_mask]
        out_np &= active_mask[:, None]

        # Hex bitmask (pure NumPy)
        self._last_hex[:] = (out_np.astype(np.int64) * self._pow2_np).sum(axis=1).astype(np.uint16)

        # _out_t shares memory with _out_np — no copy needed
        return self._out_t

    def step_sparse(
        self,
        l1_spikes_np: np.ndarray,
        dn_fired: np.ndarray,
    ) -> np.ndarray:
        """
        Sparse DEC step exploiting WTA output structure.

        Because ``BatchedTemplateLayer`` uses WTA, ``l1_spikes_np`` has
        **at most one True per row** (channel).  Instead of a full
        ``[C, n_l1] @ [C, n_l1, n_unit]`` matmul, we find the (at most C)
        firing positions with ``np.nonzero`` and do a **single row lookup**
        into ``W_np`` per channel — direct memory access to exactly the
        relevant weights, touching nothing else.

        Parameters
        ----------
        l1_spikes_np : [C, n_l1] bool  (shared-memory view from template._spk_np)
        dn_fired     : [C] bool

        Returns
        -------
        _out_np : [C, 16] bool (shared memory with _out_t)
        """
        self.t += 1
        out_np = self._out_np
        out_np[:] = False

        l1b = l1_spikes_np[:, : self.n_l1]  # [C, n_l1] — zero-copy slice

        # Delay expansion
        if self.use_delays:
            self._delay_buf_np[:, 1:] = self._delay_buf_np[:, :-1]
            self._delay_buf_np[:, 0]  = l1b
            stacked = self._delay_buf_np[:, ::-1, :].reshape(self.C, self.n_input)
        else:
            stacked = l1b  # [C, n_input]

        # DN countdown — in-place update, zero heap allocations
        wn = self._dn_window_samples
        np.subtract(self._dn_countdown_np, 1, out=self._dn_countdown_np)
        np.maximum(self._dn_countdown_np, 0, out=self._dn_countdown_np)
        if dn_fired.any():
            self._dn_countdown_np[dn_fired] = wn

        np.greater(self._dn_countdown_np, 0, out=self._active_mask_np)  # no alloc
        active_mask = self._active_mask_np
        if not active_mask.any():
            self._last_hex[:] = 0
            return self._out_np

        # [C, n_input] pre-mask — in-place AND (no [C, n_input] allocation)
        np.multiply(active_mask[:, np.newaxis], stacked, out=self._pre_mask_np)
        pre_mask = self._pre_mask_np
        self._last_pre_np[pre_mask] = self.t

        # ---- Neuron 0: any-fire — pre-alloc buffers, no astype allocation ----
        pre_mask.any(axis=1, out=self._any_bool_buf)                        # bool [C]
        np.copyto(self._any_f32_buf, self._any_bool_buf, casting='unsafe')  # bool→f32
        any_current = self._any_f32_buf
        self.mem_any_np *= self.beta
        self.mem_any_np += any_current
        np.greater_equal(self.mem_any_np, self.any_fire_threshold, out=self._spk0_buf)
        spk0 = self._spk0_buf
        self.mem_any_np[spk0] = 0.0
        out_np[:, 0] = spk0

        # ---- Neurons 1–15: sparse row lookup (WTA → ≤1 active per channel) ----
        cur_unit = self._cur_unit
        cur_unit[:] = 0.0
        # np.nonzero on [C, n_input] returns at most C pairs for WTA output
        ch_idx, ni_idx = np.nonzero(pre_mask)
        for i in range(len(ch_idx)):
            ch, ni = int(ch_idx[i]), int(ni_idx[i])
            # Direct memory access: fetch exactly one row of W_np per active spike
            cur_unit[ch] += self.W_np[ch, ni, :]  # [n_unit] — single row, no sum needed

        if self.wi_factor > 1.0:
            mc = cur_unit.mean(axis=1, keepdims=True)  # [C, 1]
            cur_unit -= mc
            cur_unit *= self.wi_factor
            cur_unit += mc

        self.mem_unit_np *= self.beta
        self.mem_unit_np += cur_unit

        winners_u = self.mem_unit_np.argmax(axis=1)
        any_above_u = self.mem_unit_np[np.arange(self.C), winners_u] >= self.unit_threshold
        spk_u = self._spk_u_np
        spk_u[:] = False
        if any_above_u.any():
            firing_ch = np.where(any_above_u)[0]
            spk_u[firing_ch, winners_u[firing_ch]] = True
        self.mem_unit_np[spk_u] = 0.0

        for ch in np.where(active_mask & spk_u.any(axis=1))[0]:
            w_idx = int(winners_u[ch])
            if (self.t - self._last_post_np[ch, w_idx]) <= self.refractory:
                spk_u[ch, w_idx] = False
            else:
                self._last_post_np[ch, w_idx] = self.t
                if not self.freeze:
                    self._stdp(ch, w_idx)

        out_np[active_mask, 1:] = spk_u[active_mask]
        out_np &= active_mask[:, None]
        self._last_hex[:] = (out_np.astype(np.int64) * self._pow2_np).sum(axis=1).astype(np.uint16)
        return self._out_np  # shared memory with _out_t

    def _stdp(self, ch: int, winner: int) -> None:
        """NumPy STDP for one channel's winning unit neuron."""
        w = self.W_np[ch, :, winner]
        w += self.ltd
        dt = self.t - self._last_pre_np[ch]
        causal = (dt <= self.ltp_win) & (dt > 0)
        w[causal] += self.ltp
        np.clip(w, self.w_lo, self.w_hi, out=w)


# ---------------------------------------------------------------------------
#  ConvergenceLayer — cross-channel ensemble state learning
# ---------------------------------------------------------------------------
class ConvergenceLayer:
    """
    Learns recurring multi-channel firing patterns via competitive STDP.

    Receives flattened DEC (or L1) spikes from all C channels and maps
    them to N_state neurons, each representing a learned ensemble state.

    Input shape:  ``[C * n_per_ch]``  (flattened spike vector)
    Output shape: ``[n_state]``  (which state neurons fired)
    """

    def __init__(
        self,
        cfg: Config,
        n_input: int,
        device: torch.device,
        *,
        n_state_neurons: int | None = None,
    ) -> None:
        cc = cfg.convergence
        self.n = int(n_state_neurons if n_state_neurons is not None else cc.n_global_neurons)
        self.n_input = n_input
        self.device = device

        tm = cc.tm_samples

        self.threshold = float(cc.threshold_factor * n_input)
        self.refractory = cc.refractory_samples
        self.freeze = cc.freeze_stdp
        self.wi_factor = cc.wi_factor

        self.ltp = cc.stdp_ltp
        self.ltp_win = cc.stdp_ltp_window
        self.ltd = cc.stdp_ltd
        self.w_lo = cc.w_lo
        self.w_hi = cc.w_hi

        # Weights [n_input, n_state] — kept as NumPy for fast CPU matmul (tiny tensors)
        rng = np.random.default_rng(seed=31)
        self.W_np = rng.uniform(cc.init_w_min, cc.init_w_max, (n_input, self.n)).astype(np.float32)
        # Keep a torch view for callers that expect a Tensor (read-only)
        self.W = torch.from_numpy(self.W_np)

        self.beta = float(np.exp(-1.0 / tm))
        self.mem_np = np.zeros(self.n, dtype=np.float32)

        self._last_pre = np.full(n_input, -9999, dtype=np.int64)
        self._last_post = np.full(self.n, -9999, dtype=np.int64)
        self.t: int = 0
        # Pre-allocated output — _spk_t shares memory with _spk_np (no copy_ per step)
        self._spk_np = np.zeros(self.n, dtype=bool)
        self._spk_t = torch.from_numpy(self._spk_np)  # shared memory: writes to _spk_np visible here
        # Pre-allocated float32 input buffer (avoids astype allocation every step)
        self._x_f32 = np.zeros(n_input, dtype=np.float32)
        # Pre-allocated scratch: active mask, wi_factor buffer, STDP causal mask
        self._active   = np.zeros(n_input, dtype=bool)         # replaces `active = x_np > 0.5`
        self._cur_wi   = np.zeros(self.n, dtype=np.float32)    # replaces `current = mc + ...`
        self._cur      = np.zeros(self.n, dtype=np.float32)    # matmul output (pre-alloc)
        self._causal   = np.zeros(n_input, dtype=bool)         # STDP causal mask

    def step(self, flat_spikes) -> torch.Tensor:
        """
        Forward one timestep.

        Parameters
        ----------
        flat_spikes : ndarray [n_input] bool/float, or Tensor on CPU
            Flattened spike vector from all channels.

        Returns
        -------
        spikes : Tensor [n_state] bool  (shared-memory view of _spk_np)
        """
        self.t += 1

        # Fill pre-allocated float32 buffer — avoids astype allocation
        x_np = self._x_f32
        if isinstance(flat_spikes, np.ndarray):
            np.copyto(x_np, flat_spikes, casting='unsafe')  # bool→float32 in C, no alloc
        else:
            x_np[:] = flat_spikes.detach().cpu().numpy()  # already float32 on CPU

        # Pre-spike timing for STDP — in-place comparison, no alloc
        np.greater(x_np, 0.5, out=self._active)
        self._last_pre[self._active] = self.t

        # Input current — write into pre-allocated buffer
        np.dot(x_np, self.W_np, out=self._cur)   # [n_state] — in-place dot, no alloc
        current = self._cur

        if self.wi_factor > 1.0:
            mc = float(current.mean())
            # In-place: cur_wi = mc + (current - mc) * wi_factor
            np.subtract(current, mc, out=self._cur_wi)
            self._cur_wi *= self.wi_factor
            self._cur_wi += mc
            current = self._cur_wi  # alias; no new array created

        # LIF membrane update
        self.mem_np *= self.beta
        self.mem_np += current

        # WTA: single winner
        winner = int(np.argmax(self.mem_np))
        self._spk_np[:] = False
        if self.mem_np[winner] >= self.threshold:
            # Refractory check
            if (self.t - self._last_post[winner]) > self.refractory:
                self._spk_np[winner] = True
                self.mem_np[winner] = 0.0
                self._last_post[winner] = self.t
                if not self.freeze:
                    self._stdp(winner)

        # _spk_t shares memory with _spk_np via torch.from_numpy — no copy needed
        return self._spk_t

    def _stdp(self, winner: int) -> None:
        w = self.W_np[:, winner]
        w += self.ltd
        dt = self.t - self._last_pre
        # In-place causal mask computation — no new array
        causal = self._causal
        np.less_equal(dt, self.ltp_win, out=causal)
        causal &= dt > 0
        w[causal] += self.ltp
        np.clip(w, self.w_lo, self.w_hi, out=w)
