# Optimization Rules — Enforceable Contract
<!--
  PURPOSE: Every rule here is derived from real performance wins measured on
  Jetson Orin (Tegra / integrated GPU, CUDA 12.6, Python 3.10).
  Any code change to the hot path MUST comply with all applicable rules.
  CI/grep enforcement patterns are provided under each rule.
-->

## Platform Facts (Jetson Orin)

| Fact | Implication |
|------|-------------|
| Integrated GPU — CPU and GPU share physical RAM | `torch.from_numpy()` and `np.frombuffer(tensor.numpy())` are **zero-copy** (no PCIe transfer). Use them to share state. |
| CUDA kernel launch overhead ≈ 5–20 µs per call | For tensors ≤ [16, 300] any CUDA dispatch is slower than CPU NumPy BLAS. Never dispatch to CUDA for tiny tensors. |
| Python function-call overhead ≈ 200 k calls/s ceiling | Per-sample Python loops are the primary bottleneck above 912 steps/s. Temporal batching + Numba JIT removes this ceiling. |
| OpenBLAS on Cortex-A78AE (8-core ARM) | NumPy einsum and matmul use BLAS threading. A single `np.einsum('ca,can->cn', ...)` over `[16,300,110]` outperforms 16 Python `.sum()` calls by ~10×. |

---

## Rule 1 — No heap allocation inside `step*()` methods

**Banned:**
```python
# FORBIDDEN — allocates every call
np.zeros((C, n))
np.ones(C)
np.array([...])
x.copy()
x.astype(np.float32)           # without out= parameter
torch.zeros(...)               # inside any step method
torch.from_numpy(x)            # inside any step method
torch.cat([...])               # inside any step method
```

**Required:**
```python
# CORRECT — pre-allocate in __init__, reuse in step*()
# __init__:
self._buf = np.zeros((C, n), dtype=np.float32)
self._x_f32 = np.zeros(C, dtype=np.float32)
self._spk_t = torch.from_numpy(self._spk_np)   # one-time shared-memory view

# step*():
self._buf[:] = 0.0                               # in-place reset
np.copyto(self._x_f32, x_bool, casting='unsafe') # bool→float32, no alloc
np.einsum('ca,can->cn', aff, W, out=self._cur)   # out= avoids temp alloc
```

**Grep enforcement:**
```bash
# Should return 0 results in any step* method body:
grep -n "np\.zeros\|np\.ones\|np\.array\|\.copy()\|\.astype(" src/snn_agent/core/batched.py
grep -n "np\.zeros\|np\.ones\|np\.array\|\.copy()\|\.astype(" src/snn_agent/core/multichannel.py
```

**Files in scope:** `batched.py`, `multichannel.py`, `encoder.py`, `attention.py`, `noise_gate.py`, `inhibition.py`, `decoder.py`

---

## Rule 2 — No snnTorch in the hot batched path

**Banned:**
```python
import snntorch as snn          # in any batched.py class
snn.Leaky(...)
snn.synaptic.Alpha(...)
```

**Required:**
- Inline LIF: `mem *= beta; mem += current`
- WTA: `winner = mem.argmax(axis=1); spk[firing_ch, winners[firing_ch]] = True`
- STDP: plain NumPy boolean index writes

**Status (2025-07-11):**
- ✅ `batched.py` — snnTorch fully removed; all LIF/WTA is inline NumPy
- ⚠️ `attention.py` — snnTorch present in single-channel legacy `AttentionNeuron` class (NOT used in hot path; `BatchedAttentionNeuron` is used)
- ⚠️ `template.py` — snnTorch present in single-channel legacy `TemplateLayer` class (NOT used in hot path)
- ⚠️ `pipeline.py` — imports snnTorch for single-channel mode only

**Grep enforcement:**
```bash
grep -n "import snntorch\|snn\." src/snn_agent/core/batched.py  # must return 0 results
```

---

## Rule 3 — No CUDA dispatch for tiny tensors

**Banned:**
```python
tensor.to('cuda')              # in any step*() method
torch.cuda.synchronize()       # in hot path
x.detach().cpu().numpy()       # when x is already on CPU
```

**Required:**
- All computation on CPU NumPy: `np.einsum`, `np.matmul`, boolean indexing
- Shared-memory views at init time only:
  ```python
  # __init__:
  self._spk_np = np.zeros((C, n), dtype=bool)
  self._spk_t = torch.from_numpy(self._spk_np)   # shared memory — zero-copy on Jetson
  # step*():
  self._spk_np[:] = False   # write numpy → automatically visible through _spk_t
  # No copy_, no .to(), no .detach()
  ```

**Why:** On Jetson Orin (integrated GPU), `torch.from_numpy` creates a view into the same physical page — writes to the NumPy array are immediately visible in the Tensor and vice versa. No copy is ever needed.

**GPU usage boundary:** CUDA is acceptable for preprocessing (`sosfilt` bulk filter for thousands of samples) but not for per-step inference.

**Grep enforcement:**
```bash
grep -n "\.to('cuda')\|\.to(device)\|cuda()\|copy_()\|\.cpu()\|\.detach()" src/snn_agent/core/batched.py
```

---

## Rule 4 — No Python `for ch in range(C)` loops in hot path

**Banned:**
```python
for ch in range(C):            # in step*() method
    noise_gates[ch].step(...)
    inhibitors[ch].gate(...)
    decoders[ch].step(...)
```

**Required — Option A: Vectorize with boolean indexing:**
```python
# Pre-allocate [C] buffers; use np.where, boolean index writes
ng_values = self._ng_values_buf
for ch in range(C):            # OK only if called < 1000× total (calibration)
    ...
# In hot path: vectorized
np.einsum('ca,can->cn', aff_f32, W, out=cur)   # replaces C matmul calls
self._last_pre_np[aff_bool] = t                 # replaces C for-loops
```

**Required — Option B: Numba `prange` for parallel channels:**
```python
@njit(parallel=True, cache=True)
def _lif_wta_block(cur_block, mem, spk_out, ...):
    N, C, n = cur_block.shape
    for t in range(N):           # sequential over time (state dependency)
        for ch in prange(C):     # parallel over channels (independent)
            ...
```

**Status:** `multichannel.step_full()` still has 4 Python `for ch in range(C)` loops. These are eliminated by `step_block()` + Numba (todo #6).

---

## Rule 5 — No running `sum(deque)` for O(n) accumulation

**Banned:**
```python
confidence = sum(self._dn_buf) / len(self._dn_buf)   # O(n) per step
```

**Required — O(1) running sum:**
```python
# __init__:
self._dn_buf = deque(maxlen=win)
self._dn_sum: int = 0

# step():
evicted = self._dn_buf[0] if len(self._dn_buf) == self._dn_buf.maxlen else False
self._dn_buf.append(dn_spike)
self._dn_sum += int(dn_spike) - int(evicted)
confidence = self._dn_sum / len(self._dn_buf)
```

**Status:** ✅ Fixed in `decoder.py` `ControlDecoder`.

---

## Rule 6 — Temporal batching: prefer `step_block(N)` over N×`step(1)`

**The ceiling problem:** Python function-call overhead is ~200 k calls/second. At 20 kHz × 16 channels, even an empty per-sample Python function causes 320 k calls/s, already above the ceiling.

**Required — block API:**
```python
# CORRECT: one Python call per N samples
result_list = channel_bank.step_block(raw_chunk)   # raw_chunk [raw_N, C]

# FORBIDDEN in steady-state (post-calibration) server loop:
for row in decimated:
    result = channel_bank.step_full(afferents, row)
```

**Block size guidelines:**

| Use case | N (block size) | Latency | Throughput |
|----------|---------------|---------|------------|
| Minimal latency | 4–8 | 0.2–0.4 ms | ~5000 steps/s |
| Balanced | 20 | 1.0 ms | ~20000 steps/s |
| Max throughput | 100 | 5.0 ms | ~50000 steps/s |

**Status:** `step_block()` implemented in `multichannel.py` (todo #6).

---

## Rule 7 — `torch.from_numpy()` at init time only; never inside `step*()`

**Banned:**
```python
# FORBIDDEN — allocates a new Tensor view every call
self._spk_t.copy_(torch.from_numpy(self._spk_np))   # also does redundant copy
```

**Required:**
```python
# CORRECT — init once, share forever
# __init__:
self._spk_np = np.zeros((C, n), dtype=bool)
self._spk_t = torch.from_numpy(self._spk_np)   # permanent zero-copy view

# step*():
self._spk_np[:] = False                          # write numpy array directly
self._spk_np[firing_ch, winners[firing_ch]] = True
# _spk_t automatically reflects the change — no copy_ needed
```

**Note:** `ConvergenceLayer.step()` still has a `self._spk_t.copy_(torch.from_numpy(self._spk_np))` — this is redundant since they already share memory. Remove it.

---

## Rule 8 — `np.einsum` with `out=` and `optimize=False` for hot BLAS

**Banned:**
```python
current = np.einsum('ca,can->cn', aff, W)               # allocates result
current = np.einsum('ca,can->cn', aff, W, optimize=True) # may insert intermediates
```

**Required:**
```python
np.einsum('ca,can->cn', aff_f32, self.W_np, out=self._cur_np, optimize=False)
# - out= → writes into pre-allocated buffer, no temp allocation
# - optimize=False → no auto-contraction reordering (already optimal for this shape)
```

**For temporal block einsum:**
```python
# [N, C, max_aff] × [C, max_aff, n] → [N, C, n]
np.einsum('tca,can->tcn', aff_f32_block, self.W_np, out=self._cur_block, optimize=False)
# One BLAS call instead of N separate per-step einsum calls
```

---

## Rule 9 — Numba JIT for inner loops with scalar state

**When to use Numba:** When the computation has sequential time-dependency (LIF membrane state) but independent parallel channels, and the inner body is too small for BLAS to dominate.

**Signature contract:**
```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def _lif_wta_stdp_block(
    cur_block,      # [N, C, n] float32 — pre-computed input currents (in-place modified)
    mem,            # [C, n] float32 — LIF state (in/out)
    W,              # [C, max_aff, n] float32 — weights (modified for STDP)
    aff_bool_block, # [N, C, max_aff] bool — afferent spikes
    last_pre,       # [C, max_aff] int64 — STDP pre-spike times (in/out)
    last_post,      # [C, n] int64 — STDP post-spike times (in/out)
    spk_out,        # [N, C, n] bool — output spikes (pre-alloc)
    beta, threshold, refractory, ltp, ltd, ltp_win, w_lo, w_hi,
    t_start,
    freeze,
) -> None:
    N, C, n = spk_out.shape
    for t in range(N):              # sequential (state dependency)
        for ch in prange(C):        # parallel (independent channels)
            ...
```

**Rules:**
- `cache=True` — compile once, cache to `__pycache__`. Never recompile in production.
- `parallel=True` with `prange(C)` for channels — C=16 threads on 8 ARM cores.
- Inner loop body: no Python object creation, no `.any()`, no list comprehensions.
- All array dtypes must be exact matches (int64, float32, bool) — Numba is dtype-strict.

---

## Rule 10 — `x.any()` not `np.any(x)` in hot path

**Banned:**
```python
np.any(x)         # Python wrapper — slower than C method
```

**Required:**
```python
x.any()           # C-level method, dispatches to numpy C API directly
x.any(axis=1)     # along axis — still C-level
```

---

## Rule 11 — Never rebuild `_FLAT_MAP` at runtime; add to `config.py`

When adding a new parameter:
1. Add to the appropriate `*Config` dataclass in `config.py`
2. Add flat-key entry to `_FLAT_MAP`: `"section_param_name": ("SectionConfig", "param_name")`
3. Never use string parsing or `getattr(cfg, ...)` chains in hot path

---

## Summary: File-Level Enforcement Table

| File | Rule 1 (no alloc) | Rule 2 (no snntorch) | Rule 3 (no CUDA) | Rule 4 (no ch-loops) | Rule 6 (block API) | Rule 7 (from_numpy once) |
|------|:-----------------:|:--------------------:|:----------------:|:--------------------:|:------------------:|:------------------------:|
| `batched.py` | ✅ | ✅ | ✅ | ✅ (template/dec) | ✅ step_sparse_block | ✅ |
| `multichannel.py` | ⚠️ (step_full) | ✅ | ✅ | ⚠️ 4 loops remain | ✅ step_block | ✅ |
| `encoder.py` | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| `attention.py` | ✅ (batched path) | ⚠️ legacy only | ✅ | ✅ (batched path) | N/A | ✅ |
| `noise_gate.py` | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| `inhibition.py` | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| `decoder.py` | ⚠️ _step_rate | ✅ | ✅ | N/A | N/A | N/A |
| `_numba_kernels.py` | ✅ | ✅ | ✅ | ✅ prange | ✅ block only | ✅ |

Legend: ✅ compliant · ⚠️ partial / in-progress · ❌ violation

---

## Checklist for New Components

Before merging any new `step*()` method, verify:

- [ ] No `np.zeros/ones/array/copy/astype` inside method body
- [ ] No `torch.zeros/ones/from_numpy` inside method body
- [ ] No `for ch in range(C)` in steady-state path
- [ ] All output arrays pre-allocated in `__init__` with correct dtype and shape
- [ ] `torch.from_numpy()` called once in `__init__`; never in `step*()`
- [ ] `np.einsum` uses `out=` and `optimize=False`
- [ ] Any per-step O(n) accumulation replaced with O(1) running sum
- [ ] `step_block(N)` implemented alongside `step()` for any new layer
- [ ] Numba `@njit(cache=True)` for inner scalar-state loops

---

*Last updated: 2025-07-11 · Baseline: 912 steps/s on Jetson Orin 16-channel · Target: 20,000 steps/s via temporal batching + Numba*
