# AGENTS.md — Machine-Readable Context Map for AI Agents
<!--
  PURPOSE: This file helps coding agents (Copilot, Cursor, Cline, Aider, etc.)
  quickly locate the right files, understand the architecture, and make changes
  without breaking the pipeline. READ THIS FIRST before modifying the codebase.
-->

## Quick Start
```
uv venv && source .venv/bin/activate
uv pip install -e ".[eval,dev]"
snn-serve                    # launch live server (http://localhost:8080)
snn-serve --mode synthetic   # start with synthetic signal
snn-evaluate                 # offline accuracy benchmark (single scenario)
snn-optimize                 # Optuna TPE hyperparameter search (Stage 1)
snn-genetic                  # genetic crossover optimizer (Stage 2)
```

## Architecture Overview
```
raw electrode signal (80 kHz UDP / LSL / synthetic)
  │
  ▼
┌──────────────────┐
│  Preprocessor    │  bandpass 300–6000 Hz + decimate ÷4 → 20 kHz
│  preprocessor.py │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  SpikeEncoder    │  temporal receptive field → boolean afferent vector
│  encoder.py      │  calibrates noise via MAD during first 8000 samples
└────────┬─────────┘
         │ afferents [n_centres × window_depth]
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌──────────────────┐                   ┌──────────────────┐
│ AttentionNeuron  │  excitatory gate  │  NoiseGateNeuron │  inhibitory gate
│ attention.py     │  (DN — fires on   │  noise_gate.py   │  (Kalman filter
│                  │   outlier energy)  │                  │   suppresses noise)
└────────┬─────────┘                   └────────┬─────────┘
         │ dn_spike (bool)                      │ suppression_factor (0–1)
         │                                      │
         ├──────────────────────────────────────┤
         ▼                                      ▼
┌──────────────────────────────────────────────────────┐
│  GlobalInhibitor  │  5 ms post-spike blanking        │
│  inhibition.py    │  blocks weak input after any L1  │
│                   │  spike; strong signals pass       │
└─────────┬────────────────────────────────────────────┘
          ▼
┌──────────────────┐
│  TemplateLayer   │  110 LIF neurons + WTA + competitive STDP
│  template.py     │  learns spike waveform templates
└────────┬─────────┘
         │ l1_spikes [n_neurons]
         ▼
┌──────────────────┐  (default: enabled, gated by use_dec flag)
│  DECLayer        │  16 LIF neurons: neuron 0 = any-fire OR gate,
│  dec_layer.py    │  neurons 1–15 learn unit identities via STDP
└────────┬─────────┘  DN-gated, optional delay expansion
         │ dec_spikes [16] + hex bitmask (uint16)
         ▼
┌──────────────────┐
│ ControlDecoder   │  rate / population / trigger → (control, confidence)
│ decoder.py       │  sends UDP hex bitmask to experiment hardware
└──────────────────┘
```

## File → Responsibility Map

| File | Purpose | Key Config | Agent Notes |
|------|---------|------------|-------------|
| `src/snn_agent/config.py` | **Single source of truth** for all parameters | All `*Config` dataclasses | Frozen dataclasses. Use `Config.from_flat()` for Optuna. Add new params here FIRST. |
| `src/snn_agent/core/preprocessor.py` | Bandpass + decimation | `PreprocessConfig` | Streaming-safe IIR via SOS. Changes here affect effective sample rate. |
| `src/snn_agent/core/encoder.py` | Temporal RF encoder | `EncoderConfig` | Calibrates on first 8000 samples. `n_afferents` is only known after calibration. |
| `src/snn_agent/core/attention.py` | DN: energy/outlier detector | `DNConfig` | Single LIF + pRel depression. Fires → excites L1 + boosts decoder confidence. |
| `src/snn_agent/core/noise_gate.py` | Kalman noise suppressor | `NoiseGateConfig` | Parallel to DN. Suppresses L1 input when signal variance ≈ noise baseline. |
| `src/snn_agent/core/inhibition.py` | Global post-spike inhibition | `InhibitionConfig` | 5 ms blanking after any L1 spike. Strong signals bypass. |
| `src/snn_agent/core/template.py` | L1 template matching + STDP | `L1Config` | snnTorch `Leaky` with WTA. Weight matrix `W` shape: `[n_afferents, n_neurons]`. |
| `src/snn_agent/core/dec_layer.py` | DEC spiking decoder (16 neurons) | `DECConfig` | Neuron 0 = DN-gated any-fire, neurons 1–15 = competitive STDP unit learners. Hex bitmask output. |
| `src/snn_agent/core/decoder.py` | Control signal generation | `DecoderConfig` | Three strategies. Confidence from DN sliding window. |
| `src/snn_agent/core/pipeline.py` | Factory: builds full chain | — | Two-phase: `build_pipeline()` (pre-calibration) → `complete_pipeline()` (post-calibration). |
| `src/snn_agent/server/app.py` | Asyncio server + WebSocket | ports, broadcast_every | Three input modes: electrode/lsl/synthetic. WebSocket broadcasts pipeline state. |
| `src/snn_agent/server/static/index.html` | Browser GUI | — | Canvas2D + D3.js network viz. Sliders send WebSocket commands. |
| `src/snn_agent/eval/evaluate.py` | Offline benchmark + multi-scenario eval | — | `evaluate_pipeline()` single run; `multi_evaluate()` averages over 4 scenarios with train/test split. |
| `src/snn_agent/eval/optimize.py` | Stage 1: Optuna TPE search | — | Reads `docs/optimization_manifest.yaml`. Uses `multi_evaluate()` with F₀.₅ objective. |
| `src/snn_agent/eval/genetic.py` | Stage 2: Genetic crossover optimizer | — | Breeds top-K trials via block-level crossover + mutation. Uses same `multi_evaluate()`. |
| `docs/optimization_manifest.yaml` | Search space + eval config | — | 17 tunable params, 4 evaluation scenarios, F₀.₅ metric, train/test split. |
| `docs/optimization_guide.md` | **Full optimization reference** | — | Methodology, metrics, gene blocks, interpretation, extension guide. |
| `docs/annet_architecture.yaml` | Original ANNet design doc | — | 818-line reference covering MATLAB→Python porting decisions. |
| `docs/scientific_principles.md` | **Full math + pseudocode + audit** | — | Merged scientific reference: LaTeX, pseudocode, plain English, 18 audited claims. |
| `docs/manifesto.json` | Machine-readable project contract | — | I/O protocols, file roles, extension points. |
| `data/best_config.json` | Current best hyperparameters | — | Updated by optimizer runs. Contains params, score, and all metrics. |

## Remote / embedded (Jetson)

- Browser: open `http://<host>:8080`; WS uses the page host (not hard-coded
  `localhost`). CLI: `--http-port`, `--ws-port` if defaults are busy.
- Multichannel: `broadcast_max_hz_mc` caps JSON broadcast rate; preprocessor
  uses vectorized `sosfilt` across channels in `ChannelBank.step_preprocess_chunk`.
- DEC path: `BatchedDECLayer` keeps delay state on GPU; avoid per-step full
  `l1.cpu().numpy()` except where the CPU decoder needs NumPy.

## Key Conventions for Agents

1. **Config is frozen** — never mutate. Use `cfg.with_overrides()` or `Config.from_flat()`.
2. **Flat-key mapping** — `_FLAT_MAP` in config.py translates `"l1_n_neurons"` ↔ `L1Config.n_neurons`. Add entries for new params.
3. **Pipeline is two-phase** — encoder must calibrate before downstream stages can be built (they need `n_afferents`).
4. **WebSocket commands** — browser sends JSON `{"key": value}` → `ws_handler()` in app.py dispatches. Add new command handlers there.
   - **Parameter tuning:** `dn_threshold`, `l1_stdp_ltp`, `l1_stdp_ltd`, `inh_duration_ms`, `inh_strength_threshold`, `ng_inhibit_below_sd`, `decoder_strategy`, `ttl_width_ms`, `ttl_high`
   - **Source launching:** `launch_synthetic` (dict with optional `duration_s`, `num_units`, `noise_level`), `launch_file` (string path to .ncs), `get_status`, `list_files`
   - **Responses:** Server replies with `{"status": "ok", "mode": ...}` or `{"status": "error", "message": ...}`, and broadcasts `{"mode_change": {"mode": ..., "state": ...}}` to all clients.
5. **Broadcast format** — JSON dict with `t`, `samples`, `dn_flags`, `spikes`, `control`, `confidence`. Extended fields: `noise_gate`, `inhibition_active`, `l1_membrane`, `dec_spikes`, `dec_hex`. In synthetic mode with GT: `accuracy` dict with `precision`, `recall`, `f_half`, `tp`, `fp`, `fn`, `latency_ms`, `n_gt`, `gt_progress`.
6. **Extension checklist** — when adding a new component:
   - Add its `*Config` dataclass to `config.py`
   - Add flat-key entries to `_FLAT_MAP`
   - Wire it in `pipeline.py` (`complete_pipeline()`)
   - Add WebSocket broadcast fields in `app.py` (`_process_stream()`)
   - Add GUI controls in `index.html` (controls panel and/or launcher)
   - Add search params to `docs/optimization_manifest.yaml`
   - Document in `docs/scientific_principles.md` (math + pseudocode + plain English)

## Optimization

Two-stage workflow. See `docs/optimization_guide.md` for full reference.

**Stage 1 — Optuna TPE:** `snn-optimize --n-trials 80`
**Stage 2 — Genetic:**   `snn-genetic --top-k 10 --n-offspring 160`

**Evaluation methodology:**
- 4 synthetic scenarios (varied seed, noise, unit count, firing rates)
- Train/test temporal split (STDP learns 0–15 s, scored on 15–20 s)
- `delta_time=2.0 ms` spike matching (5× tighter than original 10 ms)
- F₀.₅ objective (precision weighted 2× over recall)

**Search space:** 17 parameters across 6 groups:
- Encoder: `enc_overlap`, `enc_dvm_factor`, `enc_step_size`
- DN: `dn_threshold_factor`, `l1_dn_weight`, `dn_depression_tau`, `dn_depression_frac`
- L1/STDP: `l1_stdp_ltp`, `l1_stdp_ltd`, `l1_n_neurons`
- Inhibition: `inh_duration_ms`, `inh_strength_threshold`
- Noise gate: `ng_process_noise`, `ng_inhibit_below_sd`, `ng_suppression_factor`, `ng_ema_alpha`
- DEC: `dec_dn_window_ms`

See `docs/optimization_manifest.yaml` for ranges and types.

## Dependencies
- **Core**: numpy, scipy, torch, snntorch, websockets, pyyaml
- **Eval** (optional): spikeinterface, optuna
- **LSL** (optional): mne, mne-lsl
