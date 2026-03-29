"""
snn_agent.core.multichannel — Multi-channel wrapper for CPU-side pipeline components.

``ChannelBank`` holds C independent instances of Preprocessor, SpikeEncoder,
AttentionNeuron, NoiseGateNeuron, GlobalInhibitor, and ControlDecoder.
The batched GPU layers (BatchedTemplateLayer / BatchedDECLayer) are attached
after calibration via ``complete()``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import sosfilt

from snn_agent.config import Config
from snn_agent.core.preprocessor import Preprocessor
from snn_agent.core.encoder import SpikeEncoder
from snn_agent.core.attention import AttentionNeuron
from snn_agent.core.noise_gate import NoiseGateNeuron
from snn_agent.core.inhibition import GlobalInhibitor
from snn_agent.core.decoder import ControlDecoder
from snn_agent.core.batched import BatchedAttentionNeuron, BatchedTemplateLayer, BatchedDECLayer, ConvergenceLayer
from snn_agent.core._numba_kernels import warmup_kernels, encode_block_kernel

__all__ = ["ChannelBank"]


class ChannelBank:
    """
    Manages C independent single-channel CPU components plus batched GPU layers.

    Lifecycle:
        1. ``__init__`` — create C preprocessors + encoders
        2. ``step_preprocess`` / ``step_encode`` — run per sample until calibrated
        3. ``all_calibrated`` — check readiness
        4. ``complete`` — build downstream layers (attention, template, DEC, etc.)
        5. ``step_full`` — run the complete pipeline for one timestep
    """

    def __init__(self, cfg: Config, effective_cfg: Config, device: torch.device) -> None:
        C = cfg.n_channels
        self.C = C
        self.cfg = cfg
        self.effective_cfg = effective_cfg
        self.device = device

        self.preprocessors: list[Preprocessor] = []
        self.encoders: list[SpikeEncoder] = []
        for _ in range(C):
            self.preprocessors.append(Preprocessor(cfg))
            self.encoders.append(SpikeEncoder(effective_cfg))

        # Downstream components — populated by complete()
        self.attentions: list[AttentionNeuron] = []  # kept for API compat, unused after complete()
        self.batched_attention: BatchedAttentionNeuron | None = None
        self.noise_gates: list[NoiseGateNeuron | None] = []
        self.inhibitors: list[GlobalInhibitor | None] = []
        self.decoders: list[ControlDecoder] = []
        self.template: BatchedTemplateLayer | None = None
        self.dec_layer: BatchedDECLayer | None = None
        self.local_convergence: list[ConvergenceLayer] = []
        self.global_convergence: ConvergenceLayer | None = None

        self._completed = False
        self.max_aff: int = 0
        # Pre-allocated afferent bool buffer — filled zero-copy from encoder._active_indices
        self._aff_bool: np.ndarray | None = None
        self._aff_f32: np.ndarray | None = None
        self._gin_np: np.ndarray | None = None

        # Per-step scratch (filled by complete())
        self._ng_values: np.ndarray | None = None
        self._suppressions: np.ndarray | None = None
        self._inh_active: np.ndarray | None = None
        self._dn_f32: np.ndarray | None = None
        self._sup_f32: np.ndarray | None = None
        self._controls: np.ndarray | None = None
        self._confidences: np.ndarray | None = None
        self._hex_outputs: np.ndarray | None = None
        self._any_fired_buf: np.ndarray | None = None  # pre-alloc [C] bool

        # Block API buffers (allocated in complete(), grown in _ensure_block_buffers)
        self._aff_bool_block: np.ndarray | None = None
        self._aff_f32_block: np.ndarray | None = None
        self._dn_bool_block: np.ndarray | None = None
        self._dn_f32_block: np.ndarray | None = None
        self._sup_f32_block: np.ndarray | None = None
        self._block_N: int = 0
        self.use_parallel_kernels: bool = False   # set to True after benchmarking

        # Batch encoder state (built in complete())
        self._enc_centers: np.ndarray | None = None   # [C, max_nc] float64
        self._enc_n_centers: np.ndarray | None = None # [C] int64
        self._enc_dvm: np.ndarray | None = None       # [C] float64
        self._enc_shift_reg: np.ndarray | None = None # [C, max_nc, reg_width] bool
        self._enc_step_size: int = 1
        self._enc_twindow: int = 9
        self._enc_max_nc: int = 0

        # Per-channel tracking for inhibitor wiring
        self._any_l1_fired_prev = np.zeros(C, dtype=bool)

    @property
    def all_calibrated(self) -> bool:
        return all(enc.is_calibrated for enc in self.encoders)

    def step_preprocess(self, raw_samples: np.ndarray) -> list[float | None]:
        """
        Preprocess C raw samples (single-sample fallback).

        Returns a list of C values: the decimated sample or None if decimated away.
        """
        results: list[float | None] = []
        for ch in range(self.C):
            out = self.preprocessors[ch].step(float(raw_samples[ch]))
            results.append(out[0] if out else None)
        return results

    def step_preprocess_chunk(self, raw_chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess a chunk of raw samples for all C channels at once.

        Parameters
        ----------
        raw_chunk : ndarray [chunk_size, C]

        Returns
        -------
        ndarray [chunk_dec, C]
            Decimated output.  May be empty if chunk is smaller than one
            decimation period.
        """
        C = self.C
        x = np.ascontiguousarray(raw_chunk, dtype=np.float64)
        p0 = self.preprocessors[0]

        if p0.do_bandpass:
            zi = np.stack([p._zi for p in self.preprocessors], axis=-1)
            x, zi_out = sosfilt(p0._sos, x, zi=zi, axis=0)
            for ch, p in enumerate(self.preprocessors):
                p._zi = np.ascontiguousarray(zi_out[:, :, ch])

        cols: list[np.ndarray] = []
        min_len: int | None = None
        for ch in range(C):
            dec = self.preprocessors[ch].decimate_chunk(x[:, ch])
            cols.append(dec)
            if min_len is None or len(dec) < min_len:
                min_len = len(dec)
        if min_len is None or min_len == 0:
            return np.empty((0, C), dtype=np.float64)
        return np.column_stack([c[:min_len] for c in cols])

    def step_encode(self, decimated: list[float | None]) -> np.ndarray | None:
        """
        Encode C decimated samples (single-sample).

        Returns a view of the pre-allocated ``[C, max_aff]`` bool buffer,
        or None during calibration.  The buffer is filled using each
        encoder’s ``_active_indices`` cache — direct index writes, no
        ``astype`` conversion, no per-step allocation.
        """
        if decimated[0] is None:
            return None

        # Step each encoder; they update their _active_indices cache internally
        for ch in range(self.C):
            self.encoders[ch].step(decimated[ch])

        if not self.all_calibrated:
            return None

        if self.max_aff == 0:
            self.max_aff = max(enc.n_afferents for enc in self.encoders)

        if self._aff_bool is None:
            self._aff_bool = np.zeros((self.C, self.max_aff), dtype=bool)
            self._aff_f32 = np.zeros((self.C, self.max_aff), dtype=np.float32)

        # Fill both afferent buffers from encoder outputs in a single loop.
        # Read from _aff_out (live view) — encoders were already stepped above.
        buf = self._aff_bool
        f32 = self._aff_f32
        for ch in range(self.C):
            enc = self.encoders[ch]
            naff = enc.n_afferents
            aff = enc._aff_out          # live view — no second step call
            buf[ch, :naff] = aff        # contiguous bool slice copy
            f32[ch, :naff] = aff        # bool → float32 in C

        return buf  # persistent view — no allocation, no copy

    def step_encode_row(self, decimated_row: np.ndarray) -> np.ndarray | None:
        """
        Encode one row of decimated samples across all C channels.

        Parameters
        ----------
        decimated_row : ndarray [C]

        Returns
        -------
        ndarray [C, max_aff] bool view, or None if still calibrating.
        """
        for ch in range(self.C):
            self.encoders[ch].step(float(decimated_row[ch]))

        if not self.all_calibrated:
            return None

        if self.max_aff == 0:
            self.max_aff = max(enc.n_afferents for enc in self.encoders)

        if self._aff_bool is None:
            self._aff_bool = np.zeros((self.C, self.max_aff), dtype=bool)
            self._aff_f32 = np.zeros((self.C, self.max_aff), dtype=np.float32)

        buf = self._aff_bool
        f32 = self._aff_f32
        for ch in range(self.C):
            enc = self.encoders[ch]
            naff = enc.n_afferents
            aff = enc._aff_out          # live view — encoders already stepped above
            buf[ch, :naff] = aff
            f32[ch, :naff] = aff

        return buf

    def complete(self) -> None:
        """Build downstream layers after all encoders have calibrated."""
        if self._completed:
            return

        cfg = self.cfg
        effective_cfg = self.effective_cfg
        C = self.C

        n_affs = [enc.n_afferents for enc in self.encoders]
        self.max_aff = max(n_affs)

        # Pre-allocated bool afferent buffer (shared across step_encode + step_full)
        self._aff_bool = np.zeros((C, self.max_aff), dtype=bool)
        # Companion float32 buffer — filled zero-copy in step_encode, used by einsum
        self._aff_f32 = np.zeros((C, self.max_aff), dtype=np.float32)

        # Vectorised C-channel attention (replaces C separate AttentionNeuron instances)
        self.batched_attention = BatchedAttentionNeuron(effective_cfg, n_affs)

        self.template = BatchedTemplateLayer(cfg, n_affs, self.device)

        if cfg.use_dec:
            self.dec_layer = BatchedDECLayer(
                cfg, cfg.l1.n_neurons, C, self.device
            )

        # Tiered convergence: one LocalConv per probe, then one GlobalConv
        n_per_ch = self.dec_layer.N_DEC if self.dec_layer else cfg.l1.n_neurons
        if cfg.convergence.enabled and C > 1:
            ps = max(1, int(cfg.probe_size))
            n_probes = (C + ps - 1) // ps
            self.local_convergence = []
            for p in range(n_probes):
                ch_lo = p * ps
                ch_hi = min((p + 1) * ps, C)
                n_in_loc = (ch_hi - ch_lo) * n_per_ch
                self.local_convergence.append(
                    ConvergenceLayer(
                        cfg,
                        n_in_loc,
                        self.device,
                        n_state_neurons=cfg.convergence.n_local_neurons,
                    )
                )
            n_global_in = n_probes * cfg.convergence.n_local_neurons
            self.global_convergence = ConvergenceLayer(
                cfg,
                n_global_in,
                self.device,
                n_state_neurons=cfg.convergence.n_global_neurons,
            )
            # Pre-allocated float32 gin buffer — avoids torch.cat per step
            n_local_n = cfg.convergence.n_local_neurons
            self._gin_np = np.zeros(n_probes * n_local_n, dtype=np.float32)

        n_decoder_input = n_per_ch

        for ch in range(C):
            naff = n_affs[ch]
            self.attentions.append(AttentionNeuron(effective_cfg, naff))

            ng: NoiseGateNeuron | None = None
            if cfg.noise_gate.enabled:
                noise_sigma = self.encoders[ch].dvm / cfg.encoder.dvm_factor
                ng = NoiseGateNeuron(effective_cfg, noise_sigma)
            self.noise_gates.append(ng)

            inh: GlobalInhibitor | None = None
            if cfg.inhibition.enabled:
                inh = GlobalInhibitor(effective_cfg)
            self.inhibitors.append(inh)

            self.decoders.append(ControlDecoder(effective_cfg, n_decoder_input))

        self._completed = True
        # Cache per-channel afferent counts (avoid encoder attribute lookup in hot path)
        self._naff = np.array([enc.n_afferents for enc in self.encoders], dtype=np.intp)

        # ── Pre-allocate per-step scratch arrays (RULE-H01: no alloc in step_full) ──
        self._ng_values   = np.ones(C, dtype=np.float64)
        self._suppressions = np.ones(C, dtype=np.float64)
        self._inh_active  = np.zeros(C, dtype=bool)
        self._dn_f32      = np.zeros(C, dtype=np.float32)
        self._sup_f32     = np.zeros(C, dtype=np.float32)
        self._controls    = np.zeros(C, dtype=np.float64)
        self._confidences = np.zeros(C, dtype=np.float64)
        self._hex_outputs = np.zeros(C, dtype=np.uint16)
        self._any_fired_buf = np.zeros(C, dtype=bool)

        # ── Pre-allocate block API buffers (default 64 steps; grown on demand) ──
        self._ensure_block_buffers(64)

        # ── Warm up Numba JIT kernels (compile now, not during first inference) ──
        print("   🔥 Warming up Numba JIT kernels…", end=" ", flush=True)
        warmup_kernels(C=min(C, 2), A=min(self.max_aff, 8), L=min(cfg.l1.n_neurons, 16), N=2)
        print("done")

        # ── Build batch encoder state for Numba encode_block_kernel ──────────
        # All channels must be calibrated at this point.
        max_nc = max(enc.n_centers for enc in self.encoders)
        reg_width = self.encoders[0].step_size * self.encoders[0].twindow
        self._enc_max_nc = max_nc
        self._enc_step_size = self.encoders[0].step_size
        self._enc_twindow = self.encoders[0].twindow

        self._enc_centers   = np.zeros((C, max_nc), dtype=np.float64)
        self._enc_n_centers = np.array([enc.n_centers for enc in self.encoders], dtype=np.int64)
        self._enc_dvm       = np.array([enc.dvm for enc in self.encoders], dtype=np.float64)
        self._enc_shift_reg = np.zeros((C, max_nc, reg_width), dtype=bool)
        for ch, enc in enumerate(self.encoders):
            nc = enc.n_centers
            self._enc_centers[ch, :nc]  = enc.centers
            self._enc_shift_reg[ch, :nc, :] = enc._shift_reg_2d
        # Clear padded entries beyond max_aff in block buffers (zero-copy was init to 0)
        self._aff_bool_block[:, :, :] = False
        self._aff_f32_block[:, :, :] = 0.0

    def _ensure_block_buffers(self, N: int) -> None:
        """Grow block-API pre-allocated arrays if N exceeds current capacity."""
        if self._aff_bool_block is not None and self._block_N >= N:
            return
        block_N = max(N, 64)
        C, A = self.C, self.max_aff
        self._aff_bool_block = np.zeros((block_N, C, A), dtype=bool)
        self._aff_f32_block  = np.zeros((block_N, C, A), dtype=np.float32)
        self._dn_bool_block  = np.zeros((block_N, C), dtype=bool)
        self._dn_f32_block   = np.zeros((block_N, C), dtype=np.float32)
        self._sup_f32_block  = np.ones((block_N, C), dtype=np.float32)
        self._block_N = block_N

    def step_full(
        self,
        afferents: np.ndarray,
        decimated: list[float | None],
    ) -> dict:
        """
        Run the full downstream pipeline for one timestep across all channels.

        Parameters
        ----------
        afferents : ndarray [C, max_aff]
        decimated : list of C float values (post-preprocessing)

        Returns
        -------
        dict with per-channel results:
            dn_spikes   : ndarray [C] bool
            suppressions: ndarray [C] float
            l1_spikes   : Tensor  [C, n_l1] bool
            dec_spikes  : Tensor  [C, 16] bool or None
            controls    : ndarray [C] float
            confidences : ndarray [C] float
            hex_outputs : ndarray [C] uint16
            noise_gates : ndarray [C] float
            inhibitions : ndarray [C] bool
            l1_membrane : Tensor  [C, n_l1] float
        """
        C = self.C

        # ── Attention — single vectorized call across all C channels ───
        # afferents is [C, max_aff] bool (pre-allocated buffer from step_encode)
        dn_spikes = self.batched_attention.step_batch(afferents)  # [C] bool, no loop

        # ── Noise gate (CPU, per-channel loop — light-weight) ──────────
        ng_values = self._ng_values        # pre-alloc [C] float64
        ng_values[:] = 1.0
        for ch in range(C):
            if self.noise_gates[ch] is not None:
                ng_values[ch] = self.noise_gates[ch].step(decimated[ch])

        # ── Inhibition (CPU) — magnitude directly from NumPy, no GPU sync ──
        suppressions = self._suppressions  # pre-alloc [C] float64
        np.copyto(suppressions, ng_values)
        inh_active = self._inh_active      # pre-alloc [C] bool
        inh_active[:] = False
        mag_np = self.template._last_current_mag_np  # direct NumPy view
        any_prev = self._any_l1_fired_prev
        for ch in range(C):
            if self.inhibitors[ch] is not None:
                inh_factor = self.inhibitors[ch].gate(float(mag_np[ch]), bool(any_prev[ch]))
                suppressions[ch] *= inh_factor
                inh_active[ch] = self.inhibitors[ch].active

        # ── Template — sparse NumPy path (O(k·n) not O(max_aff·n)) ────
        # afferents IS self._aff_bool; _aff_f32 is already current from step_encode
        dn_f32 = self._dn_f32              # pre-alloc [C] float32
        sup_f32 = self._sup_f32            # pre-alloc [C] float32
        np.copyto(dn_f32, dn_spikes, casting='unsafe')     # bool → float32, no alloc
        np.copyto(sup_f32, suppressions, casting='unsafe')  # float64 → float32, no alloc
        l1_spikes_np = self.template.step_sparse(afferents, self._aff_f32, dn_f32, sup_f32)
        # l1_spikes_np is self.template._spk_np  — numpy bool [C, n_neurons]
        # self.template._spk_t shares memory → stays valid for convergence

        # ── DEC — sparse path (WTA → at most 1 active per channel) ────
        dec_spikes_np = None
        dec_spikes_t  = None
        hex_outputs   = self._hex_outputs  # pre-alloc [C] uint16
        hex_outputs[:] = 0

        if self.dec_layer is not None:
            dec_spikes_np = self.dec_layer.step_sparse(l1_spikes_np, dn_spikes)
            # dec_spikes_np is self.dec_layer._out_np  — numpy bool [C, 16]
            # self.dec_layer._out_t shares memory (shared via torch.from_numpy)
            hex_outputs[:] = self.dec_layer.hex_output
            decoder_input_np = dec_spikes_np
        else:
            decoder_input_np = l1_spikes_np

        # Track any-fired for inhibitor on next step (in-place — no alloc)
        decoder_input_np.any(axis=1, out=self._any_l1_fired_prev)

        # ── Convergence — pass numpy bool slices directly (no torch round-trip) ──
        convergence_spikes = None
        probe_convergence_spikes: list[torch.Tensor] | None = None
        global_convergence_spikes = None
        if self.global_convergence is not None and self.local_convergence:
            conv_src_np = decoder_input_np  # [C, n_out] bool numpy
            ps = max(1, int(self.cfg.probe_size))
            n_loc_n = self.local_convergence[0].n
            probe_convergence_spikes = []
            for p, loc in enumerate(self.local_convergence):
                ch_lo = p * ps
                ch_hi = min((p + 1) * ps, C)
                # Contiguous bool slice → numpy directly (no torch allocation)
                flat_bool = conv_src_np[ch_lo:ch_hi].ravel()   # [ps * n_out] bool
                loc_spk_t = loc.step(flat_bool)                 # fills loc._spk_np; returns shared _spk_t
                probe_convergence_spikes.append(loc_spk_t)
                # Fill pre-allocated gin buffer from the numpy backing array
                g0 = p * n_loc_n
                self._gin_np[g0 : g0 + n_loc_n] = loc._spk_np  # bool→float32 in C
            global_convergence_spikes = self.global_convergence.step(self._gin_np)  # numpy float32 in
            convergence_spikes = global_convergence_spikes

        # Decoders (CPU, per-channel)
        controls = self._controls      # pre-alloc [C] float64
        confidences = self._confidences  # pre-alloc [C] float64
        for ch in range(C):
            ctrl, conf = self.decoders[ch].step(
                decoder_input_np[ch], bool(dn_spikes[ch])
            )
            controls[ch] = ctrl
            confidences[ch] = conf

        return {
            "dn_spikes": dn_spikes,
            "suppressions": suppressions,
            # Shared-memory Tensor views — valid as long as the numpy buffers aren't reallocated
            "l1_spikes": self.template._spk_t,      # torch.from_numpy(_spk_np), zero-copy
            "dec_spikes": self.dec_layer._out_t if self.dec_layer is not None else None,
            "convergence_spikes": convergence_spikes,
            "probe_convergence_spikes": probe_convergence_spikes,
            "global_convergence_spikes": global_convergence_spikes,
            "controls": controls,
            "confidences": confidences,
            "hex_outputs": hex_outputs,
            "noise_gates": ng_values,
            "inhibitions": inh_active,
        }

    def step_full_block(self, dec_block: np.ndarray) -> list[dict]:
        """
        Process N decimated timesteps in a single block.

        Replaces N individual ``step_full`` calls with:
          1. One Numba JIT call for attention (N×C steps).
          2. One Numba JIT call for template LIF+WTA+STDP (N×C steps).
          3. N lightweight Python calls for noise gate, inhibition, DEC,
             convergence, and decoders (inherently sequential).

        Parameters
        ----------
        dec_block : ndarray [N, C] float64  — output of ``step_preprocess_chunk``.

        Returns
        -------
        list[dict] of N result dicts, each with the same keys as ``step_full``.
        """
        assert self._completed, "call complete() before step_full_block"
        N, C = dec_block.shape

        # ── Ensure block buffers are large enough ─────────────────────────────
        self._ensure_block_buffers(N)

        # ── Sync shift registers from individual encoders (handles transition from ──
        #    single-step path, e.g. after step_encode_row in the calibration row)
        for ch, enc in enumerate(self.encoders):
            nc = enc.n_centers
            self._enc_shift_reg[ch, :nc, :] = enc._shift_reg_2d

        # ── Encode N rows via Numba kernel (replaces N×C Python encoder.step calls) ──
        aff_bool_block = self._aff_bool_block[:N]
        aff_f32_block  = self._aff_f32_block[:N]
        aff_bool_block[:] = False
        aff_f32_block[:] = 0.0
        encode_block_kernel(
            np.ascontiguousarray(dec_block, dtype=np.float64),
            self._enc_centers,
            self._enc_n_centers,
            self._enc_dvm,
            self._enc_shift_reg,
            np.int64(self._enc_step_size),
            np.int64(self._enc_twindow),
            aff_bool_block,
            aff_f32_block,
        )
        # Sync back to individual encoders (keeps step_encode_row compatible)
        for ch, enc in enumerate(self.encoders):
            nc = enc.n_centers
            enc._shift_reg_2d[:, :] = self._enc_shift_reg[ch, :nc, :]
            enc._sample_count += N

        # ── Attention block — single Numba call (N × C steps) ────────────────
        dn_bool_block = self._dn_bool_block[:N]   # [N, C] bool — pre-alloc output
        self.batched_attention.step_block(aff_bool_block, parallel=self.use_parallel_kernels)
        # step_block writes into batched_attention._dn_block[:N] — copy to our buffer
        dn_bool_block[:] = self.batched_attention._dn_block[:N]

        # ── Noise gate + inhibition — sequential per-step (Kalman state) ─────
        sup_f32_block = self._sup_f32_block[:N]   # [N, C] float32 — pre-alloc
        dn_f32_block  = self._dn_f32_block[:N]    # [N, C] float32 — pre-alloc
        np.copyto(dn_f32_block, dn_bool_block, casting='unsafe')  # bool → f32
        mag_np = self.template._last_current_mag_np  # [C] float32 view
        for i in range(N):
            # Noise gate
            ng_i = self._ng_values  # reuse per-step scratch
            ng_i[:] = 1.0
            for ch in range(C):
                if self.noise_gates[ch] is not None:
                    ng_i[ch] = self.noise_gates[ch].step(float(dec_block[i, ch]))
            # Inhibition
            sup_i = ng_i  # suppression starts from noise gate output
            inh_i = self._inh_active
            inh_i[:] = False
            any_prev = self._any_l1_fired_prev
            for ch in range(C):
                if self.inhibitors[ch] is not None:
                    inh_factor = self.inhibitors[ch].gate(float(mag_np[ch]), bool(any_prev[ch]))
                    sup_i[ch] *= inh_factor   # in-place
                    inh_i[ch] = self.inhibitors[ch].active
            np.copyto(sup_f32_block[i], sup_i, casting='unsafe')  # float64→f32

        # ── Template block — single Numba call (N × C steps) ─────────────────
        l1_spk_block = self.template.step_sparse_block(
            aff_bool_block, aff_f32_block, dn_f32_block, sup_f32_block,
            parallel=self.use_parallel_kernels,
        )  # [N, C, n_neurons] bool — view into template._spk_block

        # ── DEC + convergence + decoders — per-step loop ──────────────────────
        results: list[dict] = []
        dec_N_DEC = self.dec_layer.N_DEC if self.dec_layer is not None else 0

        for i in range(N):
            l1_i  = l1_spk_block[i]     # [C, n_neurons] bool — view
            dn_i  = dn_bool_block[i]    # [C] bool

            # DEC step (inherently sequential — delay ring buffer has intra-step state)
            if self.dec_layer is not None:
                dec_i = self.dec_layer.step_sparse(l1_i, dn_i)   # [C, 16] bool
                hex_i = self._hex_outputs                         # pre-alloc [C] uint16
                hex_i[:] = self.dec_layer.hex_output
                dec_i_t = self.dec_layer._out_t                   # shared memory view
                decoder_in = dec_i
            else:
                dec_i   = None
                dec_i_t = None
                hex_i   = self._hex_outputs   # all-zero scratch
                hex_i[:] = 0
                decoder_in = l1_i

            decoder_in.any(axis=1, out=self._any_fired_buf)   # in-place, no alloc
            np.copyto(self._any_l1_fired_prev, self._any_fired_buf)

            # Convergence
            convergence_spikes = None
            probe_convergence_spikes: list[torch.Tensor] | None = None
            global_convergence_spikes = None
            if self.global_convergence is not None and self.local_convergence:
                ps = max(1, int(self.cfg.probe_size))
                n_loc_n = self.local_convergence[0].n
                probe_convergence_spikes = []
                for p, loc in enumerate(self.local_convergence):
                    ch_lo = p * ps
                    ch_hi = min((p + 1) * ps, C)
                    flat_bool = decoder_in[ch_lo:ch_hi].ravel()
                    loc_spk_t = loc.step(flat_bool)
                    probe_convergence_spikes.append(loc_spk_t)
                    g0 = p * n_loc_n
                    self._gin_np[g0 : g0 + n_loc_n] = loc._spk_np
                global_convergence_spikes = self.global_convergence.step(self._gin_np)
                convergence_spikes = global_convergence_spikes

            # Decoders
            controls    = self._controls
            confidences = self._confidences
            for ch in range(C):
                ctrl, conf = self.decoders[ch].step(decoder_in[ch], bool(dn_i[ch]))
                controls[ch]    = ctrl
                confidences[ch] = conf

            # Reconstruct suppressions for this step (for broadcast)
            sup_i_f64 = self._suppressions
            sup_i_f64[:] = sup_f32_block[i]

            results.append({
                "dn_spikes":   dn_i,
                "suppressions": sup_i_f64.copy(),   # copy needed: buffer reused next iter
                "l1_spikes":   torch.from_numpy(l1_i.copy()),    # copy: _spk_block reused
                "dec_spikes":  dec_i_t,
                "convergence_spikes": convergence_spikes,
                "probe_convergence_spikes": probe_convergence_spikes,
                "global_convergence_spikes": global_convergence_spikes,
                "controls":    controls.copy(),
                "confidences": confidences.copy(),
                "hex_outputs": hex_i,
                "noise_gates": self._ng_values.copy(),  # ng_values for this step
                "inhibitions": self._inh_active.copy(),
            })
        return results
