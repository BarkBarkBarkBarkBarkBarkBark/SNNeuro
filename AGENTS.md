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
snn-evaluate                 # offline accuracy benchmark
snn-optimize                 # Optuna hyperparameter search
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
┌──────────────────┐  (optional, gated by use_l2 flag)
│ ClassificationL. │  10 LIF neurons + lateral inhibition + STDP
│ output_layer.py  │  converges L1 patterns → unit identities
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ControlDecoder   │  rate / population / trigger → (control, confidence)
│ decoder.py       │  sends UDP control signal to experiment hardware
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
| `src/snn_agent/core/output_layer.py` | L2 convergence (optional) | `L2Config` | Only active when `Config.use_l2=True`. Lateral inhibition + STDP. |
| `src/snn_agent/core/decoder.py` | Control signal generation | `DecoderConfig` | Three strategies. Confidence from DN sliding window. |
| `src/snn_agent/core/pipeline.py` | Factory: builds full chain | — | Two-phase: `build_pipeline()` (pre-calibration) → `complete_pipeline()` (post-calibration). |
| `src/snn_agent/server/app.py` | Asyncio server + WebSocket | ports, broadcast_every | Three input modes: electrode/lsl/synthetic. WebSocket broadcasts pipeline state. |
| `src/snn_agent/server/static/index.html` | Browser GUI | — | Canvas2D + D3.js network viz. Sliders send WebSocket commands. |
| `src/snn_agent/eval/evaluate.py` | Offline benchmark | — | Runs pipeline sample-by-sample against SpikeInterface ground truth. |
| `src/snn_agent/eval/optimize.py` | Optuna TPE search | — | Reads `docs/optimization_manifest.yaml` for search space definition. |
| `docs/optimization_manifest.yaml` | Search space definition | — | Add new tunable parameters here to include them in Optuna search. |
| `docs/annet_architecture.yaml` | Original ANNet design doc | — | 818-line reference covering MATLAB→Python porting decisions. |
| `docs/scientific_principles.md` | **Full math + pseudocode + audit** | — | Merged scientific reference: LaTeX, pseudocode, plain English, 14 audited claims. |
| `docs/manifesto.json` | Machine-readable project contract | — | I/O protocols, file roles, extension points. |
| `data/best_config.json` | Current best hyperparameters | — | Trial 65: accuracy=0.733, 110 neurons, 81 active. |

## Key Conventions for Agents

1. **Config is frozen** — never mutate. Use `cfg.with_overrides()` or `Config.from_flat()`.
2. **Flat-key mapping** — `_FLAT_MAP` in config.py translates `"l1_n_neurons"` ↔ `L1Config.n_neurons`. Add entries for new params.
3. **Pipeline is two-phase** — encoder must calibrate before downstream stages can be built (they need `n_afferents`).
4. **WebSocket commands** — browser sends JSON `{"key": value}` → `ws_handler()` in app.py dispatches. Add new command handlers there.
   - **Parameter tuning:** `dn_threshold`, `l1_stdp_ltp`, `l1_stdp_ltd`, `inh_duration_ms`, `inh_strength_threshold`, `ng_inhibit_below_sd`, `decoder_strategy`
   - **Source launching:** `launch_synthetic` (dict with optional `duration_s`, `num_units`, `noise_level`), `launch_file` (string path to .ncs), `get_status`, `list_files`
   - **Responses:** Server replies with `{"status": "ok", "mode": ...}` or `{"status": "error", "message": ...}`, and broadcasts `{"mode_change": {"mode": ..., "state": ...}}` to all clients.
5. **Broadcast format** — JSON dict with `t`, `samples`, `dn_flags`, `spikes`, `control`, `confidence`. Extended fields: `noise_gate`, `inhibition_active`, `l1_membrane`, `l1_weights`, `l2_spikes`.
6. **Extension checklist** — when adding a new component:
   - Add its `*Config` dataclass to `config.py`
   - Add flat-key entries to `_FLAT_MAP`
   - Wire it in `pipeline.py` (`complete_pipeline()`)
   - Add WebSocket broadcast fields in `app.py` (`_process_stream()`)
   - Add GUI controls in `index.html` (controls panel and/or launcher)
   - Add search params to `docs/optimization_manifest.yaml`
   - Document in `docs/scientific_principles.md` (math + pseudocode + plain English)

## Optuna Search Space

Current: 8 parameters (DN threshold, L1 dn_weight, STDP ltp/ltd, encoder overlap/dvm_factor/step_size, L1 n_neurons).
Extended: +5 parameters (inhibition duration/strength, noise gate process/measurement noise and inhibit_below_sd, L2 enable flag).
See `docs/optimization_manifest.yaml` for ranges and types.

## Dependencies
- **Core**: numpy, scipy, torch, snntorch, websockets, pyyaml
- **Eval** (optional): spikeinterface, optuna
- **LSL** (optional): mne-lsl, neo
