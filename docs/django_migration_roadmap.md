# Django Migration Roadmap — SNN Agent Web UI

> Development plan for migrating the monolithic `index.html` SNN visualisation
> frontend into a modular Django application, with multi-channel support and
> improved waveform display timing.
>
> **Generated:** 2026-03-31  
> **Baseline:** `src/snn_agent/server/static/index.html` (1063 lines, single file)  
> **Backend:** `src/snn_agent/server/app.py` (asyncio + websockets + HTTP)

---

## Part 1 — Audit of Current UI Functions

### 1.1 Source Launcher (top bar)

| Feature | HTML Element(s) | WebSocket Command |
|---------|----------------|-------------------|
| Mode indicator (idle / running / error) | `#mode_dot`, `#mode_label` | `mode_change` broadcast |
| Load `.ncs` file from path | `#file_path` text input + `#btn_file` | `launch_file` |
| Browse `data/raw/` for `.ncs` files | `#btn_browse` | `list_files` |
| Launch synthetic recording | `#btn_synthetic` | `launch_synthetic` |
| Synthetic parameters (duration, units, noise) | `#synth_dur`, `#synth_units`, `#synth_noise` | embedded in `launch_synthetic` dict |
| Status message area | `#launcher_msg` | server response `status` / `error` |

### 1.2 Signal Visualisation Canvases

| Canvas | ID | Size | Data Source (broadcast field) | Rendering Style |
|--------|------|------|------------------------------|-----------------|
| Raw waveform (oscilloscope) | `#wave` | 900×120 | `samples` | Scrolling green line, auto-scaled, DN orange tint, inhibition red tint |
| Attention Neuron (DN) strip | `#dn_cv` | 900×16 | `dn_flags` | Orange vertical bars on DN fire |
| Noise Gate strip | `#ng_cv` | 900×16 | `noise_gate` | Magenta intensity + green level dot |
| L1 Template raster | `#raster` | 900×180 | `spikes` | Scrolling raster, 110 rows, green dots |
| DEC decoder raster (16 neurons) | `#dec_cv` | 900×48 | `dec_spikes` | Rainbow-coloured per neuron |
| Control signal trace | `#ctrl_cv` | 900×60 | `control` | Cyan dot trace around zero line |

### 1.3 Network Topology Diagram

| Feature | Implementation |
|---------|---------------|
| 4-column node layout (ENC → DN/NG/INH → L1 → DEC) | Canvas2D manual draw in `drawNetworkViz()` |
| DN node glow on fire | Shadow + fill colour |
| Noise gate suppression bar | Filled rect proportional to `noise_gate` |
| Inhibition indicator | Red glow when `inhibition_active` |
| L1 membrane heat (40 visible nodes) | Green intensity from `l1_membrane` |
| L1 spike glow | Green shadow on fire |
| DEC 16-neuron column (neuron 0 = ANY) | Rainbow + white, hex output text |
| Legend | 6-item colour key |

### 1.4 Stats Bar

| Stat | ID | Source |
|------|----|--------|
| Timestep | `#s_t` | `t` |
| Current sample value | `#s_sample` | last of `samples` |
| DN fire rate | `#s_dn` | accumulated `dn_flags` / time |
| L1 spikes/s | `#s_l1hz` | accumulated `spikes.length` / time |
| Control value | `#s_ctrl` | `control` |
| Confidence | `#s_conf` | `confidence` |
| Noise gate factor | `#s_ng` | `noise_gate` |
| Inhibition state | `#s_inh` | `inhibition_active` |
| DEC hex output | `#s_hex` | `dec_hex` |

### 1.5 Parameter Controls (sliders / selects)

| Control | ID | WebSocket Key | Range |
|---------|----|---------------|-------|
| DN Threshold | `#dn_th_slider` | `dn_threshold` | 10–300 |
| STDP LTP amplitude | `#ltp_slider` | `l1_stdp_ltp` | 0.001–0.04 |
| STDP LTD amplitude | `#ltd_slider` | `l1_stdp_ltd` | −0.02 – −0.0005 |
| Inhibition duration (ms) | `#inh_dur_slider` | `inh_duration_ms` | 0–15 |
| Inhibition strength threshold | `#inh_str_slider` | `inh_strength_threshold` | 20–400 |
| Noise gate σ threshold | `#ng_sd_slider` | `ng_inhibit_below_sd` | 0.5–5.0 |
| Decoder strategy | `#decoder_sel` | `decoder_strategy` | discrete/ttl/trigger/rate/population |
| TTL pulse width (ms) | `#ttl_width_slider` | `ttl_width_ms` | 0.1–10 (visible only when strategy=ttl) |
| TTL high level | `#ttl_high_slider` | `ttl_high` | 0.1–1.0 (visible only when strategy=ttl) |

### 1.6 WebSocket Connection

| Feature | Detail |
|---------|--------|
| Auto-connect | `ws://localhost:8765` |
| Reconnect on close | 2 s retry loop |
| Status indicator | `#status` text (● connected / ● disconnected) |
| Inbound: stream data | `samples`, `dn_flags`, `spikes`, `control`, `confidence`, `noise_gate`, `inhibition_active`, `l1_membrane`, `dec_spikes`, `dec_hex` |
| Inbound: server response | `status`, `mode_change`, `files` |
| Outbound: parameter tuning | JSON `{key: value}` for each slider/select |
| Outbound: source commands | `launch_synthetic`, `launch_file`, `list_files`, `get_status` |
| DN threshold auto-init | First broadcast with `dn_th` initialises slider |

---

## Part 2 — Known Issues to Address

### 2.1 Waveform Scrolls Too Fast

**Root cause:** Each WebSocket broadcast delivers `broadcast_every` samples
(default 5) and the render loop shifts the canvas left by `N` pixels — one
pixel per sample. At 20 kHz effective rate with `broadcast_every=5`, that's
4,000 broadcasts/s × 5 px shift = the entire 900 px canvas redraws in 225 ms.
Spikes flash by in under a millisecond of screen time.

**Goal:** The waveform should scroll at a rate where individual spike
waveforms (≈1 ms) are visible — roughly 50–200 ms of signal visible on
screen at once.

### 2.2 Single Channel Only

**Root cause:** The pipeline (`_process_stream`) processes one channel at a
time. The synthetic source generates `num_channels=1`. The UI has no concept
of channel selection. The `--channels 4` CLI flag seen in terminal history
does not exist in the current `app.py` argument parser.

**Goal:** Support N parallel pipelines (one per channel), display a channel
selector, and show per-channel stats.

### 2.3 No Input Configuration Page

**Goal:** A dedicated page where the user can configure number of channels,
select files, generate synthetic recordings, and set signal parameters before
the pipeline starts.

---

## Part 3 — Django Architecture

### 3.1 Design Principles

- **Minimum viable code** — use Django templates + vanilla JS, not a SPA
  framework
- **Maintainability over cleverness** — plain function views, no class-based
  view inheritance chains, clear template hierarchy
- **Keep the asyncio pipeline** — Django serves pages and REST endpoints;
  the existing asyncio WebSocket server runs as a parallel process
- **Feature parity first** — every function from Part 1 must exist before
  adding new features

### 3.2 Project Layout

```
SNNeuro/
├── manage.py
├── snn_web/                          # Django project
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── dashboard/                        # Main app
│   ├── __init__.py
│   ├── urls.py
│   ├── views.py                      # page views (input, monitor, about)
│   ├── api.py                        # JSON endpoints for AJAX calls
│   ├── consumers.py                  # Django Channels WebSocket consumer
│   ├── routing.py                    # WebSocket URL routing
│   ├── models.py                     # Session, Channel, Recording models
│   ├── forms.py                      # Input config forms
│   ├── templatetags/
│   │   └── snn_tags.py               # custom template filters
│   ├── templates/
│   │   └── dashboard/
│   │       ├── base.html             # shared layout (nav, footer, WS status)
│   │       ├── input.html            # input config page
│   │       ├── monitor.html          # live visualisation page
│   │       └── partials/
│   │           ├── _channel_card.html    # per-channel card (reusable)
│   │           ├── _controls.html        # parameter sliders
│   │           ├── _launcher.html        # source launcher bar
│   │           ├── _stats_bar.html       # stats readout
│   │           ├── _network_viz.html     # topology canvas
│   │           └── _waveform.html        # signal canvas stack
│   └── static/
│       └── dashboard/
│           ├── css/
│           │   └── snn.css           # extracted from <style> block
│           └── js/
│               ├── websocket.js      # WS connect/reconnect/dispatch
│               ├── waveform.js       # waveform canvas rendering
│               ├── raster.js         # L1 + DEC raster rendering
│               ├── network_viz.js    # topology diagram
│               ├── controls.js       # slider/select bindings
│               ├── stats.js          # stats bar updater
│               ├── launcher.js       # source launcher logic
│               └── channels.js       # channel card selection logic
└── src/
    └── snn_agent/                    # existing pipeline (unchanged)
        ├── config.py
        ├── core/
        ├── eval/
        ├── io/
        └── server/
            └── app.py                # still runs asyncio WS + pipeline
```

### 3.3 Technology Choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Web framework | Django 5.x | Mature, batteries included, good templates |
| WebSocket bridge | Django Channels + Daphne | Lets Django pages talk to the asyncio pipeline WS |
| Async pipeline | Keep existing `app.py` | Avoid rewrite; run as subprocess or import |
| JS framework | None (vanilla JS + ES modules) | Minimum viable, no build step, matches existing |
| CSS | Single extracted stylesheet | No preprocessor needed for this scope |
| Channel layer | Redis (or in-memory for dev) | Django Channels needs a layer for WS pub/sub |

### 3.4 Communication Architecture

```
Browser                    Django (Daphne)              Pipeline (asyncio)
  │                              │                              │
  │  HTTP GET /monitor/          │                              │
  │─────────────────────────────►│                              │
  │  ◄── rendered HTML + JS      │                              │
  │                              │                              │
  │  WS connect /ws/stream/      │                              │
  │─────────────────────────────►│                              │
  │                              │  WS connect ws://localhost:8765
  │                              │─────────────────────────────►│
  │                              │                              │
  │                              │  ◄── JSON broadcast          │
  │  ◄── forward to browser      │                              │
  │                              │                              │
  │  slider change {key: val}    │                              │
  │─────────────────────────────►│                              │
  │                              │  forward ──────────────────►│
  │                              │                              │
```

Django Channels acts as a **proxy** — it connects to the existing `app.py`
WebSocket server as a client, forwards broadcasts to all browser clients,
and relays commands back. This avoids rewriting the pipeline's event loop.

---

## Part 4 — Development Roadmap

### Phase 0: Foundation (Django project scaffold)

- [ ] **0.1** Install Django + Channels: `uv pip install django channels daphne channels-redis`
- [ ] **0.2** Create Django project: `django-admin startproject snn_web .`
- [ ] **0.3** Create dashboard app: `python manage.py startapp dashboard`
- [ ] **0.4** Configure `settings.py`: add `daphne`, `channels`, `dashboard` to `INSTALLED_APPS`; set `ASGI_APPLICATION`, `CHANNEL_LAYERS` (in-memory for dev), static files
- [ ] **0.5** Create `base.html` template with dark theme CSS (extracted from index.html `<style>` block), nav bar (Input | Monitor), WS status indicator
- [ ] **0.6** Add `snn-web` entry point to `pyproject.toml` scripts
- [ ] **0.7** Verify bare Django boots: `python manage.py runserver` → see base template

### Phase 1: Input Page (new feature)

- [ ] **1.1** Create `InputConfigForm` in `forms.py`: number of channels (1–32), source type (synthetic / file / electrode / LSL), per-source params
- [ ] **1.2** Create `input.html` template: channel count selector, source type tabs, synthetic params (duration, units, noise, seed), file browser, electrode/LSL config
- [ ] **1.3** Create `views.input_page` view: render form, accept POST → store config in session
- [ ] **1.4** Add API endpoint `api.launch_source` (POST): accepts config JSON, starts pipeline via subprocess or import, returns status
- [ ] **1.5** Add API endpoint `api.list_files` (GET): returns `.ncs` files from `data/raw/`
- [ ] **1.6** Wire URL routes: `/` → input page, `/api/launch/` → launch, `/api/files/` → file list
- [ ] **1.7** Test: can configure 4-channel synthetic, click launch, get redirect to monitor page

### Phase 2: Monitor Page — Feature Parity (canvas migration)

- [ ] **2.1** Extract CSS from `index.html` → `static/dashboard/css/snn.css`
- [ ] **2.2** Extract JS into modules:
  - [ ] `websocket.js` — connect, reconnect, dispatch incoming messages
  - [ ] `waveform.js` — raw signal canvas rendering
  - [ ] `raster.js` — L1 raster + DEC raster canvases
  - [ ] `network_viz.js` — topology diagram canvas
  - [ ] `controls.js` — slider/select event bindings + `sendCmd()`
  - [ ] `stats.js` — stats bar updater
  - [ ] `launcher.js` — source launcher bar (keep as fallback on monitor page)
- [ ] **2.3** Create `monitor.html` template using `{% include %}` partials:
  - [ ] `_launcher.html` — source launcher bar
  - [ ] `_waveform.html` — waveform + DN + NG canvases
  - [ ] `_stats_bar.html` — stats readout
  - [ ] `_controls.html` — parameter sliders/selects
  - [ ] `_network_viz.html` — topology canvas
- [ ] **2.4** Create Django Channels `StreamConsumer` in `consumers.py`:
  - Connects to pipeline WS (`ws://localhost:8765`) as a client on `connect()`
  - Forwards all inbound pipeline messages to browser via `self.send()`
  - Forwards browser commands to pipeline WS
  - Handles disconnect/reconnect
- [ ] **2.5** Wire `routing.py`: `ws/stream/` → `StreamConsumer`
- [ ] **2.6** Smoke test: load `/monitor/`, verify all 6 canvases render, all 9 sliders work, launcher works, stats update
- [ ] **2.7** Parity checklist — verify each item from Part 1:
  - [ ] Source launcher with mode dot, file load, browse, synthetic launch
  - [ ] Waveform canvas with DN tint, inhibition tint, auto-scale
  - [ ] DN strip
  - [ ] Noise gate strip
  - [ ] L1 raster (110 neurons, grid lines)
  - [ ] DEC raster (16 neurons, rainbow colours)
  - [ ] Control signal trace
  - [ ] Network topology (all 4 columns, all indicators)
  - [ ] Stats bar (all 9 values)
  - [ ] All 9 parameter controls
  - [ ] WebSocket auto-reconnect
  - [ ] DN threshold auto-initialisation from first broadcast

### Phase 3: Fix Waveform Scroll Speed

- [ ] **3.1** Add a `time_window_ms` setting to the waveform renderer (default: 100 ms)
- [ ] **3.2** Instead of 1 px per sample, calculate: `px_per_sample = canvas_width / (time_window_ms * effective_fs / 1000)`
- [ ] **3.3** Accumulate incoming samples into a ring buffer (size = `time_window_ms × effective_fs / 1000`)
- [ ] **3.4** On each `requestAnimationFrame`, redraw the entire visible window from the ring buffer — no canvas shifting
- [ ] **3.5** Add a "Time window" slider to `_controls.html` (range: 10 ms – 1000 ms, default: 100 ms) — client-side only, no WS command needed
- [ ] **3.6** Apply the same ring-buffer approach to DN strip, noise gate strip, and control signal canvas for consistency
- [ ] **3.7** Test: at 20 kHz effective rate, 100 ms window → 2000 samples across 900 px → individual spike waveforms (~20 samples = ~9 px) are clearly visible

### Phase 4: Multi-Channel Support

#### 4.1 Backend (pipeline changes)

- [ ] **4.1.1** Add `num_channels: int = 1` to `SyntheticConfig` and `Config`
- [ ] **4.1.2** Add flat-key `synth_num_channels` to `_FLAT_MAP`
- [ ] **4.1.3** Modify `_synthetic_source()` in `app.py`: generate multi-channel recording (`num_channels=N`), yield `(frame_idx, channel_idx, sample)` tuples
- [ ] **4.1.4** Modify `_process_stream()`: maintain `N` parallel pipelines (one `Preprocessor + Encoder + AttentionNeuron + ... + Decoder` per channel)
- [ ] **4.1.5** Extend broadcast JSON: add `"channel"` field to each broadcast, or broadcast all channels in one message: `{"channels": {0: {...}, 1: {...}, ...}}`
- [ ] **4.1.6** Add `--channels` CLI arg to `snn-serve`
- [ ] **4.1.7** Test: `snn-serve --mode synthetic --channels 4` → WS broadcasts include 4 channels of data

#### 4.2 Frontend (channel cards)

- [ ] **4.2.1** Create `_channel_card.html` partial: compact card showing channel number, mode dot (active/idle), mini waveform thumbnail, DN rate, L1 spike rate, control value
- [ ] **4.2.2** Create `channels.js`: render channel cards from broadcast data, handle click-to-select, highlight active channel
- [ ] **4.2.3** Add channel card strip to `monitor.html` (horizontal scrollable row above the main canvas area)
- [ ] **4.2.4** When a channel card is clicked, the main canvas stack (waveform, DN, NG, L1, DEC, control, network) switches to display that channel's data
- [ ] **4.2.5** Channel card shows a live mini-waveform (tiny canvas, ~200×40) updated at reduced rate (every 10th broadcast)
- [ ] **4.2.6** Stats bar shows channel number and per-channel stats
- [ ] **4.2.7** Test: 4-channel synthetic → 4 cards visible → click card 3 → main display shows channel 3 data

### Phase 5: Polish & Integration

- [ ] **5.1** Add Django `manage.py` command `run_all` that starts both Django (Daphne) and the pipeline (`snn-serve`) in parallel
- [ ] **5.2** Update `pyproject.toml` with Django dependencies in a new `[project.optional-dependencies] web` extra
- [ ] **5.3** Add recording session model: store launch config, start time, channel count (useful for future replay)
- [ ] **5.4** Add error handling: if pipeline WS is not running, show clear message on monitor page with "Start Pipeline" button
- [ ] **5.5** Mobile-responsive CSS: channel cards wrap, canvases scale to viewport width
- [ ] **5.6** Add favicon and page titles per page
- [ ] **5.7** Update `AGENTS.md` with new file→responsibility entries for all Django files
- [ ] **5.8** Update `docs/manifesto.json` with new architecture

---

## Part 5 — Verification Matrix

Every feature from the original `index.html` must be present in the Django
version. Use this matrix during testing:

| # | Original Feature | Django Location | Phase | Status |
|---|-----------------|-----------------|-------|--------|
| 1 | WS auto-connect + reconnect | `websocket.js` | 2 | ☐ |
| 2 | Mode dot + label | `_launcher.html` + `launcher.js` | 2 | ☐ |
| 3 | File path input + Load | `_launcher.html` + `launcher.js` | 2 | ☐ |
| 4 | Browse `.ncs` files | `_launcher.html` + `launcher.js` | 2 | ☐ |
| 5 | Synthetic launch + params | `_launcher.html` + `launcher.js` | 2 | ☐ |
| 6 | Status message area | `_launcher.html` + `launcher.js` | 2 | ☐ |
| 7 | Raw waveform canvas (green line) | `_waveform.html` + `waveform.js` | 2 | ☐ |
| 8 | Waveform DN orange tint | `waveform.js` | 2 | ☐ |
| 9 | Waveform inhibition red tint | `waveform.js` | 2 | ☐ |
| 10 | Waveform auto-scale | `waveform.js` | 2 | ☐ |
| 11 | DN tick strip | `_waveform.html` + `waveform.js` | 2 | ☐ |
| 12 | Noise gate strip | `_waveform.html` + `waveform.js` | 2 | ☐ |
| 13 | L1 raster (110 neurons) | `_waveform.html` + `raster.js` | 2 | ☐ |
| 14 | L1 raster grid lines | `raster.js` | 2 | ☐ |
| 15 | DEC raster (16 neurons, rainbow) | `_waveform.html` + `raster.js` | 2 | ☐ |
| 16 | Control signal trace | `_waveform.html` + `waveform.js` | 2 | ☐ |
| 17 | Network topology (4-col layout) | `_network_viz.html` + `network_viz.js` | 2 | ☐ |
| 18 | Network DN glow | `network_viz.js` | 2 | ☐ |
| 19 | Network NG suppression bar | `network_viz.js` | 2 | ☐ |
| 20 | Network INH indicator | `network_viz.js` | 2 | ☐ |
| 21 | Network L1 membrane heat + spikes | `network_viz.js` | 2 | ☐ |
| 22 | Network DEC column (16 neurons) | `network_viz.js` | 2 | ☐ |
| 23 | Network hex output text | `network_viz.js` | 2 | ☐ |
| 24 | Network legend | `network_viz.js` | 2 | ☐ |
| 25 | Stats: timestep | `_stats_bar.html` + `stats.js` | 2 | ☐ |
| 26 | Stats: sample value | `stats.js` | 2 | ☐ |
| 27 | Stats: DN fire rate | `stats.js` | 2 | ☐ |
| 28 | Stats: L1 spikes/s | `stats.js` | 2 | ☐ |
| 29 | Stats: control value | `stats.js` | 2 | ☐ |
| 30 | Stats: confidence | `stats.js` | 2 | ☐ |
| 31 | Stats: noise gate factor | `stats.js` | 2 | ☐ |
| 32 | Stats: inhibition state | `stats.js` | 2 | ☐ |
| 33 | Stats: DEC hex output | `stats.js` | 2 | ☐ |
| 34 | Slider: DN threshold | `_controls.html` + `controls.js` | 2 | ☐ |
| 35 | Slider: STDP LTP | `controls.js` | 2 | ☐ |
| 36 | Slider: STDP LTD | `controls.js` | 2 | ☐ |
| 37 | Slider: inhibition duration | `controls.js` | 2 | ☐ |
| 38 | Slider: inhibition strength | `controls.js` | 2 | ☐ |
| 39 | Slider: noise gate σ | `controls.js` | 2 | ☐ |
| 40 | Select: decoder strategy | `controls.js` | 2 | ☐ |
| 41 | Slider: TTL width (conditional) | `controls.js` | 2 | ☐ |
| 42 | Slider: TTL level (conditional) | `controls.js` | 2 | ☐ |
| 43 | DN threshold auto-init from broadcast | `controls.js` | 2 | ☐ |
| 44 | TTL rows hide/show on strategy change | `controls.js` | 2 | ☐ |
| — | **New features** | | | |
| 45 | Waveform time-window control | `waveform.js` + `_controls.html` | 3 | ☐ |
| 46 | Ring-buffer waveform rendering | `waveform.js` | 3 | ☐ |
| 47 | Input configuration page | `input.html` + `views.py` | 1 | ☐ |
| 48 | Multi-channel pipeline | `app.py` + `config.py` | 4 | ☐ |
| 49 | Channel cards with mini-waveforms | `_channel_card.html` + `channels.js` | 4 | ☐ |
| 50 | Channel switching on main display | `channels.js` + all renderers | 4 | ☐ |

---

## Part 6 — Implementation Notes

### 6.1 Waveform Ring Buffer (Phase 3 detail)

```javascript
// waveform.js — ring buffer approach
const RING_SIZE = 4000;            // 200 ms at 20 kHz
const ring = new Float32Array(RING_SIZE);
let ringHead = 0;

function pushSamples(samples) {
  for (const s of samples) {
    ring[ringHead % RING_SIZE] = s;
    ringHead++;
  }
}

function renderWaveform(ctx, width, height, windowSamples) {
  ctx.clearRect(0, 0, width, height);
  const start = Math.max(0, ringHead - windowSamples);
  const pxPerSample = width / windowSamples;
  // ... draw from ring[start] to ring[ringHead]
}
```

The key insight: **don't shift the canvas** (`drawImage(cv, -N, 0)`).
Instead, keep a fixed-size buffer and redraw the visible window each frame.
This decouples scroll speed from broadcast rate.

### 6.2 Multi-Channel Broadcast Format (Phase 4 detail)

Option A — per-channel messages (simpler, backward compatible):
```json
{"channel": 0, "t": 12345, "samples": [...], "spikes": [...], ...}
{"channel": 1, "t": 12345, "samples": [...], "spikes": [...], ...}
```

Option B — bundled (fewer messages, but larger):
```json
{
  "t": 12345,
  "channels": {
    "0": {"samples": [...], "spikes": [...], ...},
    "1": {"samples": [...], "spikes": [...], ...}
  }
}
```

**Recommendation:** Option A. Simpler to implement, the existing single-channel
message format is unchanged (just add a `"channel"` field defaulting to 0),
and the consumer can route by channel ID.

### 6.3 Django Channels Consumer Skeleton

```python
# dashboard/consumers.py
import asyncio
import json
import websockets
from channels.generic.websocket import AsyncWebsocketConsumer

PIPELINE_WS = "ws://localhost:8765"

class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self._pipeline_ws = await websockets.connect(PIPELINE_WS)
        self._reader_task = asyncio.create_task(self._read_pipeline())

    async def _read_pipeline(self):
        try:
            async for msg in self._pipeline_ws:
                await self.send(text_data=msg)  # forward to browser
        except websockets.ConnectionClosed:
            await self.close()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data and self._pipeline_ws:
            await self._pipeline_ws.send(text_data)  # forward to pipeline

    async def disconnect(self, code):
        if hasattr(self, '_reader_task'):
            self._reader_task.cancel()
        if hasattr(self, '_pipeline_ws'):
            await self._pipeline_ws.close()
```

### 6.4 Preserving the Existing Server

The existing `app.py` asyncio server (`snn-serve`) continues to run
independently. Django does **not** replace it — it wraps it. This means:

1. `snn-serve` runs on ports 8765 (WS) and 8080 (HTTP, can be disabled)
2. Django runs on port 8000 (HTTP + WS via Daphne)
3. The browser connects to Django on port 8000
4. Django's `StreamConsumer` connects to `snn-serve` on port 8765

This architecture lets us migrate incrementally. The old `index.html` at
`localhost:8000` (Django dashboard).

---

## Part 7 — Dependency Additions

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
web = [
    "django>=5.0",
    "channels>=4.0",
    "daphne>=4.0",
    "channels-redis>=4.0",  # swap for channels[in-memory] in dev
]
```

---

## Part 8 — File Checklist for Agents

When implementing each phase, create or modify these files:

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 0 | `manage.py`, `snn_web/{settings,urls,wsgi,asgi}.py`, `dashboard/{__init__,urls,views,models,forms}.py`, `dashboard/templates/dashboard/base.html`, `dashboard/static/dashboard/css/snn.css` | `pyproject.toml` |
| 1 | `dashboard/forms.py`, `dashboard/api.py`, `dashboard/templates/dashboard/input.html` | `dashboard/urls.py`, `dashboard/views.py` |
| 2 | `dashboard/consumers.py`, `dashboard/routing.py`, `dashboard/templates/dashboard/monitor.html`, all `_partials/*.html`, all `static/dashboard/js/*.js` | `dashboard/urls.py`, `snn_web/urls.py` |
| 3 | — | `static/dashboard/js/waveform.js`, `dashboard/templates/dashboard/partials/_controls.html` |
| 4 | `dashboard/templates/dashboard/partials/_channel_card.html`, `static/dashboard/js/channels.js` | `src/snn_agent/config.py`, `src/snn_agent/server/app.py`, `src/snn_agent/core/pipeline.py`, `static/dashboard/js/websocket.js` |
| 5 | — | `AGENTS.md`, `docs/manifesto.json`, `pyproject.toml` |
