# AGENTS.md — Machine-Readable Context Map for AI Agents
<!--
  PURPOSE: This file helps coding agents (Copilot, Cursor, Cline, Aider, etc.)
  quickly locate the right files, understand the architecture, and make changes
  without breaking the pipeline. READ THIS FIRST before modifying the codebase.
-->

## Quick Start
```
uv venv && source .venv/bin/activate
uv pip install -e ".[eval,dev,web]"

# ── Browser dashboard (primary entry point) ───────────────────────────────────
./start.sh                                        # synthetic mode, 1 channel
./start.sh --channels 4 --mode synthetic --config data/a_best_config.json
# or via Django management:
python manage.py run_all --mode synthetic --channels 4
# Browse from your machine: http://<server-ip>:8000/

# ── Pipeline server only (no browser UI) ─────────────────────────────────────
snn-serve                                         # WebSocket on ws://localhost:8765
snn-serve --mode synthetic --channels 4 --config data/a_best_config.json

# ── Evaluation / optimisation ─────────────────────────────────────────────────
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

<<<<<<< HEAD
### Django Web UI (multi-page dashboard)

| File | Purpose | Agent Notes |
|------|---------|-------------|
| `start.sh` | **Primary launch script** for SSH deployments | Starts `snn-serve` + `daphne`, prints `http://<LAN-IP>:8000/`. Use this from the command line. |
| `manage.py` | Django management entry point | `python manage.py run_all` also starts both servers. |
| `snn_web/settings.py` | Django settings | `PIPELINE_WS_URL = "ws://localhost:8765"`. No database; no sessions. `DEBUG=True` serves statics via `ASGIStaticFilesHandler`. |
| `snn_web/asgi.py` | ASGI app — HTTP + WebSocket routing | Wraps HTTP with `ASGIStaticFilesHandler` in DEBUG; WS uses `AllowedHostsOriginValidator`. |
| `snn_web/urls.py` | Root URL conf | Delegates all routes to `dashboard.urls`. |
| `snn_web/management.py` | Entry point for `snn-web` CLI script | Calls `django.core.management.execute_from_command_line`. |
| `dashboard/urls.py` | Dashboard URL patterns | `/` → input, `/monitor/` → monitor, `/api/launch/`, `/api/files/`, `/api/status/`. |
| `dashboard/views.py` | Page views | Sessionless: config passed as URL query params (`?channels=N&source=synthetic`). |
| `dashboard/forms.py` | `InputConfigForm` | `num_channels` (1–32), `source_type` radio, synthetic/file params, decoder strategy. |
| `dashboard/api.py` | JSON API endpoints | Forwards commands to pipeline via short-lived WS connection. Uses `asyncio.new_event_loop()` in sync views. |
| `dashboard/consumers.py` | `StreamConsumer` — WS proxy | Forwards browser ↔ pipeline WS at `PIPELINE_WS_URL`. Sends friendly error if pipeline not running. |
| `dashboard/routing.py` | WS URL routing | Maps `ws/stream/` → `StreamConsumer`. |
| `dashboard/static/dashboard/css/snn.css` | All dashboard styles | Extracted from `index.html` + nav, channel cards, input page styles. |
| `dashboard/static/dashboard/js/websocket.js` | WS connection manager | `onMessage(type, fn)` dispatcher. Types: `stream`, `mode_change`, `files`, `status`, `open`, `*`. |
| `dashboard/static/dashboard/js/waveform.js` | Ring-buffer waveform renderer | Fixes scroll-speed bug. `setTimeWindow(ms)`, `setChannel(ch)`. `EFFECTIVE_FS=20000` hardcoded. |
| `dashboard/static/dashboard/js/raster.js` | L1 + DEC raster renderer | Ring-buffer spike events. Syncs time window with waveform. |
| `dashboard/static/dashboard/js/network_viz.js` | Network topology diagram | Throttled ~15fps. Ported from `index.html`. |
| `dashboard/static/dashboard/js/stats.js` | Stats bar 1Hz updater | Reads `msg.channel` to filter by active channel. |
| `dashboard/static/dashboard/js/controls.js` | Parameter sliders + time window | `initDNThreshold(val)` from first broadcast. Time window is client-only (no WS command). |
| `dashboard/static/dashboard/js/launcher.js` | Source launcher bar | Handles file/synthetic launch buttons, mode dot, file browse. |
| `dashboard/static/dashboard/js/channels.js` | Channel card strip | Mini-waveforms per channel, click-to-select. Routes by `msg.channel ?? 0`. |
| `dashboard/templates/dashboard/base.html` | Base layout | Nav with INPUT/MONITOR links, WS status indicator. |
| `dashboard/templates/dashboard/input.html` | Input config page | Source type tabs, channel count, synthetic/file params, decoder strategy. |
| `dashboard/templates/dashboard/monitor.html` | Live monitor page | ES module imports, `NUM_CHANNELS` from Django context. |
| `dashboard/templates/dashboard/partials/` | Template partials | `_launcher`, `_waveform`, `_stats_bar`, `_network_viz`, `_controls`. |
| `dashboard/management/commands/run_all.py` | `manage.py run_all` command | Starts both servers as subprocesses; handles Ctrl+C cleanup. |
=======
## Remote / embedded (Jetson)

- Browser: open `http://<host>:8080`; WS uses the page host (not hard-coded
  `localhost`). CLI: `--http-port`, `--ws-port` if defaults are busy.
- Multichannel: `broadcast_max_hz_mc` caps JSON broadcast rate; preprocessor
  uses vectorized `sosfilt` across channels in `ChannelBank.step_preprocess_chunk`.
- DEC path: `BatchedDECLayer` keeps delay state on GPU; avoid per-step full
  `l1.cpu().numpy()` except where the CPU decoder needs NumPy.
>>>>>>> 316d4a9 (running fast parallel, still low observability and no output from the final layer)

## Key Conventions for Agents

1. **Config is frozen** — never mutate. Use `cfg.with_overrides()` or `Config.from_flat()`.
2. **Flat-key mapping** — `_FLAT_MAP` in config.py translates `"l1_n_neurons"` ↔ `L1Config.n_neurons`. Add entries for new params.
3. **Pipeline is two-phase** — encoder must calibrate before downstream stages can be built (they need `n_afferents`).
4. **WebSocket commands** — browser sends JSON `{"key": value}` → `ws_handler()` in app.py dispatches. Add new command handlers there.
   - **Parameter tuning:** `dn_threshold`, `l1_stdp_ltp`, `l1_stdp_ltd`, `inh_duration_ms`, `inh_strength_threshold`, `ng_inhibit_below_sd`, `decoder_strategy`, `ttl_width_ms`, `ttl_high`
   - **Source launching:** `launch_synthetic` (dict with optional `duration_s`, `num_units`, `noise_level`), `launch_file` (string path to .ncs), `get_status`, `list_files`
   - **Responses:** Server replies with `{"status": "ok", "mode": ...}` or `{"status": "error", "message": ...}`, and broadcasts `{"mode_change": {"mode": ..., "state": ...}}` to all clients.
5. **Broadcast format** — JSON dict with `t`, `channel`, `samples`, `dn_flags`, `spikes`, `control`, `confidence`. Extended fields: `noise_gate`, `inhibition_active`, `l1_membrane`, `dec_spikes`, `dec_hex`. The `channel` field (0-indexed) is present in all messages; JS renderers filter by `msg.channel ?? 0`.
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
- **Web** (optional): django>=5.0, channels>=4.0, daphne>=4.0
