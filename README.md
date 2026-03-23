# ⚡ SNN Agent — Real-time Spike Sorting for Intracranial Recordings

**Turn raw electrode noise into sorted neural spikes — live, in your browser.**

<p align="center">
  <img src="docs/assets/live_raster.png" alt="Live spike raster" width="900">
</p>

SNN Agent is a streaming spiking neural network that watches your electrode
signal and learns to detect, classify, and sort action potentials in real
time.  No training data required.  No manual template selection.  Plug in
your signal and watch it learn.

## Why SNN Agent?

| Problem | SNN Agent's Solution |
|---|---|
| Offline spike sorting delays your experiment by hours | Sorts spikes **in real time** as they arrive |
| Traditional sorters need hand-picked templates | The network **learns templates automatically** via competitive plasticity (STDP) |
| Noise floods your signal with false positives | A **Kalman-filter noise gate** and **post-spike inhibition** suppress noise while letting real spikes through |
| Closed-loop experiments need instant feedback | A **control decoder** converts sorted spikes into a control signal with sub-millisecond latency |
| Setup is complicated | One command to install, one command to run. **Browser GUI** with live raster, network topology, and tunable parameters |

## How it works

```
electrode → filter → encode → detect → sort → decode → control signal
```

1. **Filter** — Bandpass cleans your signal (300 Hz – 6 kHz) and decimation
   drops the sample rate to a manageable 20 kHz.
2. **Encode** — Each sample is converted into a binary activation pattern —
   like a tiny image of the waveform's shape over time.
3. **Detect** — An attention neuron watches for moments when the signal
   energy rises above the noise floor.  It fires when something interesting
   happens.
4. **Sort** — 110 spiking neurons compete to match waveform templates.
   Each neuron specialises on a different spike shape through competitive
   learning.  The best match wins.
5. **Decode** — Sorted spikes are converted into a control signal you can
   send to your experiment hardware in real time.

Everything runs at 20 kHz with results streaming live to your browser.

> 📐 For the full mathematical specification, equations, pseudocode, and
> scientific basis, see **[Scientific Principles](docs/scientific_principles.md)**.

## Quick start

```bash
# 1. Install — requires uv (https://docs.astral.sh/uv/)
uv venv && uv pip install -e ".[all]"

# 2. Start the agent
snn-serve

# 3. Open your browser
open http://localhost:8080
```

From the browser GUI you can launch a **synthetic recording** or **load a
`.ncs` file** directly — no extra terminal needed.

## Input modes

SNN Agent accepts signal from three sources.  You can switch between
synthetic and file modes live from the browser.

| Mode | How to use |
|---|---|
| **Synthetic** | Click 🧪 SYNTHETIC in the browser (or `snn-serve --mode synthetic`) |
| **File (.ncs)** | Enter a file path in the browser launcher and click ▶ LOAD FILE |
| **LSL stream** | Start `snn-lsl data/raw/CSC285_0001.ncs` in a separate terminal, then `snn-serve` |
| **Electrode (UDP)** | Feed real samples to UDP port 9001: `snn-serve --mode electrode` |

### CLI commands

| Command | What it does |
|---|---|
| `snn-serve` | Start the agent server + browser GUI |
| `snn-serve --mode synthetic` | Start with built-in synthetic signal |
| `snn-lsl <ncs_path>` | Replay a `.ncs` file over Lab Streaming Layer |
| `snn-test-electrode` | Synthetic UDP test signal generator |
| `snn-evaluate` | Offline pipeline evaluation against ground truth |
| `snn-optimize` | Automated hyperparameter search (Optuna) |

## Browser GUI

The live dashboard runs at `http://localhost:8080` and shows:

- **Source launcher** — load files or start synthetic recordings without
  leaving the browser
- **Raw signal** oscilloscope with attention neuron and noise gate overlays
- **Spike raster** — 110-neuron template layer activity in real time
- **Control signal** trace with confidence indicator
- **Network topology** — animated diagram showing which neurons are firing,
  the noise gate state, and inhibition activity
- **Tunable parameters** — adjust DN threshold, STDP learning rates,
  inhibition, noise gate sensitivity, and decoder strategy live

## Architecture at a glance

```
┌─────────────────────────────────────────┐
│  Input: UDP / LSL / .ncs / synthetic    │
└──────────────┬──────────────────────────┘
               ▼
        ┌──────────────┐
        │ Preprocessor │  Bandpass 300–6 kHz + decimate
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ SpikeEncoder │  Temporal receptive field → binary vector
        └──────┬───────┘
          ┌────┴────┐
          ▼         ▼
    ┌──────────┐ ┌───────────┐
    │ Attention│ │ Noise Gate│  Excitatory + inhibitory gates
    │ Neuron   │ │ (Kalman)  │
    └────┬─────┘ └─────┬─────┘
         └──────┬──────┘
                ▼
        ┌───────────────┐
        │ Global Inhibit│  Post-spike blanking
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Template Layer│  110 LIF neurons + WTA + STDP
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │    Decoder    │  → Control signal + UDP out
        └───────┬───────┘
           ┌────┴────┐
           ▼         ▼
      Experiment   Browser
      hardware     GUI
```

## Configuration

All parameters live in frozen dataclasses in `src/snn_agent/config.py`.
Override defaults with:

```python
from snn_agent.config import Config

cfg = Config()                                    # defaults (best trial)
cfg = Config.from_flat({"l1_n_neurons": 40})      # flat keys (Optuna)
cfg = cfg.with_overrides(sampling_rate_hz=30000)  # keyword overrides
```

## Hyperparameter optimisation

Automated search using [Optuna](https://optuna.org/) (TPE sampler) against
SpikeInterface ground-truth synthetic recordings:

```bash
snn-optimize                    # default trial budget
snn-optimize --n-trials 80     # custom budget
snn-evaluate                    # single evaluation run
```

Results are saved to `data/best_config.json` and `data/trials.csv`.
See `docs/optimization_manifest.yaml` for the full search space definition.

## Project structure

```
snn-agent/
├── pyproject.toml                  # Package definition & entry points
├── README.md                       # ← You are here
├── src/snn_agent/
│   ├── config.py                   # All parameters (frozen dataclasses)
│   ├── core/
│   │   ├── preprocessor.py         # Bandpass + decimation
│   │   ├── encoder.py              # Temporal RF spike encoder
│   │   ├── attention.py            # Attention neuron (DN)
│   │   ├── noise_gate.py           # Kalman noise suppressor
│   │   ├── inhibition.py           # Global post-spike inhibition
│   │   ├── template.py             # Template layer (L1) — 110 LIF + STDP
│   │   ├── output_layer.py         # Classification layer (L2, optional)
│   │   ├── decoder.py              # Control decoder
│   │   └── pipeline.py             # Factory: builds the full chain
│   ├── server/
│   │   ├── app.py                  # Async server — WebSocket + HTTP
│   │   └── static/index.html       # Browser GUI
│   ├── eval/
│   │   ├── evaluate.py             # Offline scorer (SpikeInterface)
│   │   ├── optimize.py             # Optuna hyperparameter search
│   │   └── ground_truth.py         # Synthetic ground-truth generation
│   └── io/
│       ├── lsl_player.py           # Replay .ncs over LSL
│       └── test_electrode.py       # Synthetic UDP signal generator
├── data/
│   ├── best_config.json            # Best optimisation trial
│   ├── trials.csv                  # Full optimisation history
│   └── raw/                        # Raw neural recordings (.ncs)
└── docs/
    ├── scientific_principles.md    # Full math, pseudocode & scientific basis
    ├── optimization_manifest.yaml  # Search space definition
    ├── annet_architecture.yaml     # Original ANNet design reference
    └── manifesto.json              # Machine-readable project contract
```

## Dependencies

**Core** (installed by default):
`numpy` · `scipy` · `torch` · `snntorch` · `websockets` · `pyyaml`

| Optional group | Install with | Adds |
|---|---|---|
| `lsl` | `uv pip install -e ".[lsl]"` | MNE, MNE-LSL (Lab Streaming Layer + .ncs file reading) |
| `eval` | `uv pip install -e ".[eval]"` | SpikeInterface, Optuna (evaluation & optimisation) |
| `all` | `uv pip install -e ".[all]"` | Everything above |
| `dev` | `uv pip install -e ".[dev]"` | + pytest, ruff (development) |

## Learn more

- **[Scientific Principles](docs/scientific_principles.md)** — full
  mathematical specification, pseudocode, plain-English explanations, and
  audited scientific claims with literature references.
- **[Architecture Reference](docs/annet_architecture.yaml)** — detailed
  ANNet→Python porting decisions.
- **[Optimisation Manifest](docs/optimization_manifest.yaml)** — search
  space definition for Optuna.

## License

MIT.  See `pyproject.toml` for details.
