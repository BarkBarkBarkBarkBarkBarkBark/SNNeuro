"""
snn_agent.core._numba_kernels — Numba JIT kernels for temporal-batched SNN hot path.

All kernels process N timesteps **sequentially** (STDP requires temporal ordering)
with a plain ``range(C)`` inner loop over channels.  JIT compilation eliminates
Python dispatch overhead for the per-element operations, giving 5–10× speedup
over equivalent NumPy + Python on Jetson Orin (Cortex-A78AE) for the array
sizes used in this pipeline ([C=16, A≈300, L=110]).

Naming conventions
------------------
    N  = block size (timesteps)
    C  = channels
    A  = max_aff (padded afferent count per channel)
    L  = n_neurons (L1 / template neurons per channel)

Parallel variant (prange)
--------------------------
Functions suffixed ``_parallel`` use ``prange(C)`` to parallelize over
channels within each timestep.  On Jetson Orin (8 ARM cores, OpenMP thread
pool) this is beneficial when C ≥ 8 and N ≥ 4.  Use the non-parallel
versions when C < 8 to avoid synchronisation overhead.

Cache note
----------
Compiled bitcode is cached in ``~/.cache/numba/``.  First call compiles;
subsequent calls load from cache.  If you change any kernel signature or body,
delete the cache directory or set ``cache=False`` temporarily.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
#  Attention (DN) block
# ---------------------------------------------------------------------------

@njit(cache=True)
def attention_block(
    aff_bool_block,   # [N, C, A] bool
    exp_td,           # [max_dt+1] float32 — precomputed exponential LUT
    p_rel,            # [C, A] float32   — synaptic depression state (in-place)
    last_pre,         # [C, A] int64     — last pre-spike time (in-place)
    v,                # [C]    float64   — membrane potential (in-place)
    dn_out,           # [N, C] bool      — output: DN spikes (in-place)
    t_start,          # int64            — timestep at start of block
    decay,            # float64          — membrane decay factor
    threshold,        # float64          — firing threshold
    reset,            # float64          — soft-reset membrane level
    f_d,              # float32          — depression fraction
    max_dt,           # int64            — LUT clip value
) -> None:
    """
    Vectorised attention (DN) neuron block.

    Sequential over N (LIF state dependency), plain loop over C.
    Modifies ``p_rel``, ``last_pre``, ``v``, and ``dn_out`` in-place.
    """
    N, C, A = aff_bool_block.shape
    for ti in range(N):
        t = t_start + ti + 1
        for ch in range(C):
            contrib = 0.0
            for k in range(A):
                if aff_bool_block[ti, ch, k]:
                    dt = t - last_pre[ch, k]
                    if dt > max_dt:
                        dt = max_dt
                    # Synaptic recovery: p_rec = 1 − (1 − p_rel) · exp(−Δt/τ)
                    p_rec_k = 1.0 - (1.0 - p_rel[ch, k]) * exp_td[dt]
                    contrib += p_rec_k
                    # Depression: p_rel ← p_rec · (1 − f_d)
                    p_rel[ch, k] = p_rec_k * (1.0 - f_d)
                    last_pre[ch, k] = t
            # LIF membrane update
            v[ch] = v[ch] * decay + contrib
            if v[ch] >= threshold:
                v[ch] = reset
                dn_out[ti, ch] = True
            else:
                dn_out[ti, ch] = False


@njit(parallel=True, cache=True)
def attention_block_parallel(
    aff_bool_block,   # [N, C, A] bool
    exp_td,           # [max_dt+1] float32
    p_rel,            # [C, A] float32
    last_pre,         # [C, A] int64
    v,                # [C]    float64
    dn_out,           # [N, C] bool
    t_start,          # int64
    decay,            # float64
    threshold,        # float64
    reset,            # float64
    f_d,              # float32
    max_dt,           # int64
) -> None:
    """
    Parallel variant: ``prange(C)`` within each timestep.

    Safe because each channel has fully independent state arrays
    (p_rel[ch,*], last_pre[ch,*], v[ch]).  No cross-channel reads/writes.
    Use when C ≥ 8 and profiling shows parallel benefit on the target platform.
    """
    N, C, A = aff_bool_block.shape
    for ti in range(N):
        t = t_start + ti + 1
        for ch in prange(C):
            contrib = 0.0
            for k in range(A):
                if aff_bool_block[ti, ch, k]:
                    dt = t - last_pre[ch, k]
                    if dt > max_dt:
                        dt = max_dt
                    p_rec_k = 1.0 - (1.0 - p_rel[ch, k]) * exp_td[dt]
                    contrib += p_rec_k
                    p_rel[ch, k] = p_rec_k * (1.0 - f_d)
                    last_pre[ch, k] = t
            v[ch] = v[ch] * decay + contrib
            if v[ch] >= threshold:
                v[ch] = reset
                dn_out[ti, ch] = True
            else:
                dn_out[ti, ch] = False


# ---------------------------------------------------------------------------
#  Template LIF + WTA + competitive STDP block
# ---------------------------------------------------------------------------

@njit(cache=True)
def template_lif_wta_block(
    aff_bool_block,  # [N, C, A] bool
    aff_f32_block,   # [N, C, A] float32
    W,               # [C, A, L] float32 — weight matrix (STDP updates in-place)
    mem,             # [C, L] float32    — membrane potential (in-place)
    last_pre,        # [C, A] int64      — last pre-spike time (in-place)
    last_post,       # [C, L] int64      — last post-spike time (in-place)
    spk_out,         # [N, C, L] bool    — spike output (in-place)
    cur_scratch,     # [C, L] float32    — scratch buffer (pre-allocated, no heap)
    last_mag,        # [C]    float32    — pre-suppression current magnitude (in-place)
    dn_block,        # [N, C] float32    — DN spike amplitudes
    sup_block,       # [N, C] float32    — suppression factors
    t_start,         # int64
    beta,            # float32           — membrane decay factor
    threshold,       # float32           — WTA firing threshold
    refractory,      # int64             — refractory period (samples)
    dn_weight,       # float32           — DN→current excitatory weight
    freeze,          # bool              — if True, STDP is disabled
    ltp,             # float32           — STDP LTP increment
    ltd,             # float32           — STDP LTD decrement
    ltp_win,         # int64             — STDP LTP causal window (samples)
    w_lo,            # float32           — weight lower bound
    w_hi,            # float32           — weight upper bound
) -> None:
    """
    Template layer block: N timesteps × C channels.

    Sequential outer loop over N (STDP requires temporal ordering).
    Inner loop over C is plain range — see template_lif_wta_block_parallel
    for the prange variant.

    All state arrays (W, mem, last_pre, last_post) are modified in-place.
    ``spk_out`` is fully overwritten (set False then True for winners).
    ``cur_scratch`` is a caller-provided scratch buffer, not heap-allocated.
    """
    N, C, A = aff_bool_block.shape
    L = W.shape[2]

    for ti in range(N):
        t = t_start + ti + 1

        for ch in range(C):
            # ── 1. Clear current scratch for this (timestep, channel) ──────
            for j in range(L):
                cur_scratch[ch, j] = 0.0

            # ── 2. Input current: sum W[ch, k, :] for each active afferent ─
            #       Also update pre-spike timing for STDP.
            for k in range(A):
                if aff_bool_block[ti, ch, k]:
                    last_pre[ch, k] = t
                    af = aff_f32_block[ti, ch, k]
                    for j in range(L):
                        cur_scratch[ch, j] += W[ch, k, j] * af

            # ── 3. DN excitatory boost ──────────────────────────────────────
            dn_val = dn_block[ti, ch]
            if dn_val > 0.0:
                dn_c = dn_weight * dn_val
                for j in range(L):
                    cur_scratch[ch, j] += dn_c

            # ── 4. Pre-suppression magnitude (for inhibitor bypass logic) ───
            max_c = 0.0
            for j in range(L):
                if cur_scratch[ch, j] > max_c:
                    max_c = cur_scratch[ch, j]
            last_mag[ch] = max_c

            # ── 5. Apply suppression ────────────────────────────────────────
            sf = sup_block[ti, ch]
            for j in range(L):
                cur_scratch[ch, j] *= sf

            # ── 6. LIF membrane update + WTA argmax ─────────────────────────
            best_val = -1e38
            winner = 0
            for j in range(L):
                v = mem[ch, j] * beta + cur_scratch[ch, j]
                mem[ch, j] = v
                if v > best_val:
                    best_val = v
                    winner = j

            # ── 7. Clear spike output for this (ti, ch) ─────────────────────
            for j in range(L):
                spk_out[ti, ch, j] = False

            # ── 8. Fire + soft reset + refractory + STDP ────────────────────
            if best_val >= threshold:
                if (t - last_post[ch, winner]) > refractory:
                    spk_out[ti, ch, winner] = True
                    mem[ch, winner] = 0.0
                    last_post[ch, winner] = t
                    if not freeze:
                        # Competitive STDP: LTD for all → LTP for causal pre-spikes
                        for k in range(A):
                            W[ch, k, winner] += ltd
                            dt = t - last_pre[ch, k]
                            if 0 < dt <= ltp_win:
                                W[ch, k, winner] += ltp
                            if W[ch, k, winner] < w_lo:
                                W[ch, k, winner] = w_lo
                            elif W[ch, k, winner] > w_hi:
                                W[ch, k, winner] = w_hi


@njit(parallel=True, cache=True)
def template_lif_wta_block_parallel(
    aff_bool_block,  # [N, C, A] bool
    aff_f32_block,   # [N, C, A] float32
    W,               # [C, A, L] float32
    mem,             # [C, L] float32
    last_pre,        # [C, A] int64
    last_post,       # [C, L] int64
    spk_out,         # [N, C, L] bool
    cur_scratch,     # [C, L] float32
    last_mag,        # [C]    float32
    dn_block,        # [N, C] float32
    sup_block,       # [N, C] float32
    t_start,         # int64
    beta,            # float32
    threshold,       # float32
    refractory,      # int64
    dn_weight,       # float32
    freeze,          # bool
    ltp,             # float32
    ltd,             # float32
    ltp_win,         # int64
    w_lo,            # float32
    w_hi,            # float32
) -> None:
    """
    Parallel variant using ``prange(C)`` within each timestep.

    Data-race proof: every array write is indexed by the prange variable ``ch``
    (W[ch,*,*], mem[ch,*], last_pre[ch,*], last_post[ch,*], spk_out[ti,ch,*],
    cur_scratch[ch,*], last_mag[ch]) — no two threads share a write target.

    Use when C ≥ 8 and benchmarking shows >1.5× speedup over serial variant.
    On Jetson Orin, enables up to 8-way channel parallelism.
    """
    N, C, A = aff_bool_block.shape
    L = W.shape[2]

    for ti in range(N):
        t = t_start + ti + 1
        for ch in prange(C):
            for j in range(L):
                cur_scratch[ch, j] = 0.0
            for k in range(A):
                if aff_bool_block[ti, ch, k]:
                    last_pre[ch, k] = t
                    af = aff_f32_block[ti, ch, k]
                    for j in range(L):
                        cur_scratch[ch, j] += W[ch, k, j] * af
            dn_val = dn_block[ti, ch]
            if dn_val > 0.0:
                dn_c = dn_weight * dn_val
                for j in range(L):
                    cur_scratch[ch, j] += dn_c
            max_c = 0.0
            for j in range(L):
                if cur_scratch[ch, j] > max_c:
                    max_c = cur_scratch[ch, j]
            last_mag[ch] = max_c
            sf = sup_block[ti, ch]
            for j in range(L):
                cur_scratch[ch, j] *= sf
            best_val = -1e38
            winner = 0
            for j in range(L):
                v = mem[ch, j] * beta + cur_scratch[ch, j]
                mem[ch, j] = v
                if v > best_val:
                    best_val = v
                    winner = j
            for j in range(L):
                spk_out[ti, ch, j] = False
            if best_val >= threshold:
                if (t - last_post[ch, winner]) > refractory:
                    spk_out[ti, ch, winner] = True
                    mem[ch, winner] = 0.0
                    last_post[ch, winner] = t
                    if not freeze:
                        for k in range(A):
                            W[ch, k, winner] += ltd
                            dt = t - last_pre[ch, k]
                            if 0 < dt <= ltp_win:
                                W[ch, k, winner] += ltp
                            if W[ch, k, winner] < w_lo:
                                W[ch, k, winner] = w_lo
                            elif W[ch, k, winner] > w_hi:
                                W[ch, k, winner] = w_hi


# ---------------------------------------------------------------------------
#  Temporal receptive field encoder block
# ---------------------------------------------------------------------------

@njit(cache=True)
def encode_block_kernel(
    samples,          # [N, C] float64   — raw decimated samples
    centers,          # [C, max_nc] float64  — amplitude bin centres (padded)
    n_centers,        # [C] int64        — real centre count per channel
    dvm,              # [C] float64      — half-bin width
    shift_reg,        # [C, max_nc, reg_width] bool — shift register (in-place)
    step_size,        # int64            — subsampling factor
    twindow,          # int64            — output delay taps
    aff_bool_out,     # [N, C, max_aff] bool    — output (in-place)
    aff_f32_out,      # [N, C, max_aff] float32 — output (in-place)
) -> None:
    """
    Temporal receptive field encoder for N timesteps × C channels.

    Vectorises the N×C call loop from ChannelBank.step_full_block into
    a single Numba JIT call, eliminating N×C Python function-call overheads.

    Each channel has an independent shift register (shape [max_nc, reg_width])
    that stores the last ``reg_width = step_size × twindow`` activation columns.
    The afferent output for channel ``ch`` occupies positions
    ``[0 : n_centers[ch] × twindow]`` in the output row; the rest remain False.

    Sequential over N (shift register is stateful), plain range over C.
    """
    N, C = samples.shape
    reg_width = shift_reg.shape[2]

    for ti in range(N):
        for ch in range(C):
            x = samples[ti, ch]
            nc = n_centers[ch]
            dv = dvm[ch]

            # ── In-place left shift: oldest column discarded, rightmost freed ──
            for k in range(nc):
                for j in range(reg_width - 1):
                    shift_reg[ch, k, j] = shift_reg[ch, k, j + 1]

            # ── Activate bins for this sample ────────────────────────────────
            for k in range(nc):
                diff = x - centers[ch, k]
                if diff < 0.0:
                    diff = -diff
                shift_reg[ch, k, reg_width - 1] = diff <= dv

            # ── Build afferent output (subsampled by step_size) ──────────────
            #    Afferent index = k * twindow + t
            #    Reading shift_reg column t*step_size gives the t-th delay tap.
            for k in range(nc):
                for t in range(twindow):
                    v = shift_reg[ch, k, t * step_size]
                    idx = k * twindow + t
                    aff_bool_out[ti, ch, idx] = v
                    aff_f32_out[ti, ch, idx] = 1.0 if v else 0.0


# ---------------------------------------------------------------------------
#  Warm-up helper (call once at import time after calibration)
# ---------------------------------------------------------------------------

def warmup_kernels(C: int = 2, A: int = 8, L: int = 16, N: int = 2) -> None:
    """
    Trigger Numba JIT compilation for all kernels with tiny dummy arrays.

    Call this once after calibration completes (i.e., inside
    ``ChannelBank.complete()``) so the first real inference block is not
    penalised by compilation latency (~1–3 s on Jetson Orin).
    """
    aff_b = np.zeros((N, C, A), dtype=bool)
    aff_f = np.zeros((N, C, A), dtype=np.float32)
    exp_td = np.ones(10, dtype=np.float32)
    p_rel = np.ones((C, A), dtype=np.float32)
    lp = np.full((C, A), -9999, dtype=np.int64)
    v = np.zeros(C, dtype=np.float64)
    dn_out = np.zeros((N, C), dtype=bool)
    attention_block(aff_b, exp_td, p_rel, lp, v, dn_out,
                    np.int64(0), np.float64(0.9), np.float64(1.0),
                    np.float64(0.0), np.float32(0.1), np.int64(9))
    attention_block_parallel(aff_b, exp_td, p_rel.copy(), lp.copy(), v.copy(),
                             np.zeros((N, C), dtype=bool),
                             np.int64(0), np.float64(0.9), np.float64(1.0),
                             np.float64(0.0), np.float32(0.1), np.int64(9))

    W = np.zeros((C, A, L), dtype=np.float32)
    mem = np.zeros((C, L), dtype=np.float32)
    lp2 = np.full((C, A), -9999, dtype=np.int64)
    lpost = np.full((C, L), -9999, dtype=np.int64)
    spk_out = np.zeros((N, C, L), dtype=bool)
    cur_sc = np.zeros((C, L), dtype=np.float32)
    lmag = np.zeros(C, dtype=np.float32)
    dn_f = np.zeros((N, C), dtype=np.float32)
    sup_f = np.ones((N, C), dtype=np.float32)
    template_lif_wta_block(
        aff_b, aff_f, W, mem, lp2, lpost, spk_out, cur_sc, lmag, dn_f, sup_f,
        np.int64(0), np.float32(0.9), np.float32(1.0), np.int64(5),
        np.float32(1.0), False, np.float32(0.05), np.float32(-0.01),
        np.int64(20), np.float32(0.0), np.float32(1.0),
    )
    template_lif_wta_block_parallel(
        aff_b, aff_f, W.copy(), mem.copy(), lp2.copy(), lpost.copy(),
        np.zeros((N, C, L), dtype=bool), cur_sc.copy(), lmag.copy(),
        dn_f, sup_f,
        np.int64(0), np.float32(0.9), np.float32(1.0), np.int64(5),
        np.float32(1.0), False, np.float32(0.05), np.float32(-0.01),
        np.int64(20), np.float32(0.0), np.float32(1.0),
    )

    # Encoder kernel
    nc_arr = np.array([A], dtype=np.int64)
    centers_w = np.zeros((C, A), dtype=np.float64)
    dvm_w = np.ones(C, dtype=np.float64)
    sreg_w = np.zeros((C, A, 4), dtype=bool)  # reg_width=4, step_size=1, twindow=4
    ab_out = np.zeros((N, C, A * 4), dtype=bool)
    af_out = np.zeros((N, C, A * 4), dtype=np.float32)
    samp = np.zeros((N, C), dtype=np.float64)
    # broadcast nc_arr to all C channels
    nc_full = np.full(C, A, dtype=np.int64)
    encode_block_kernel(samp, centers_w, nc_full, dvm_w, sreg_w,
                        np.int64(1), np.int64(4), ab_out, af_out)
