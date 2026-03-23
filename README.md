# ⚡ SNN Agent — Real-time Spiking Neural Network for Neural Signals

<p align="center">
  <img src="docs/assets/live_raster.png" alt="Live spike raster visualisation" width="900">
</p>

A streaming spiking neural network that detects and classifies extracellular
action potentials in real time, derived from the
[MB2018 ANNet](docs/reference/MB2018%20-%20ANNet/read_me.txt) architecture.

```
raw signal → bandpass → decimate → encode → attention neuron → template layer → control decoder
              300–6 kHz   80→20 kHz   temporal RF   LIF + pRel     110 LIF + STDP   rate/pop/trigger
```

## Quick start

```bash
# 1. Install (once) — requires uv (https://docs.astral.sh/uv/)
uv venv && uv pip install -e ".[all]"

# 2. Start the agent (default: LSL mode)
snn-serve

# 3. Open the live raster in your browser
open http://localhost:8080
```

## Run modes

| Command | What it does |
|---|---|
| `snn-serve` | Start the agent (reads `mode` from `Config`) |
| `snn-serve --mode electrode` | Start agent in electrode (UDP) mode |
| `snn-serve --mode synthetic` | Start agent with built-in synthetic signal |
| `snn-lsl <ncs_path>` | Replay a `.ncs` file over LSL |
| `snn-test-electrode` | Synthetic UDP test signal generator |
| `snn-evaluate` | Offline pipeline evaluation against ground truth |
| `snn-evaluate --overrides '{"l1_n_neurons": 40}'` | Evaluate with config overrides |
| `snn-optimize` | Hyperparameter optimisation (Optuna) |
| `snn-optimize --n-trials 80` | Custom trial budget |
| `snn-ground-truth` | Generate & inspect synthetic ground truth |

### LSL mode (replay a recording)

```bash
# Terminal 1 — stream the NCS file over Lab Streaming Layer
snn-lsl data/raw/CSC285_0001.ncs

# Terminal 2 — start the agent (auto-connects to LSL)
snn-serve
```

### Electrode mode (synthetic or live UDP)

```bash
# With the built-in synthetic test signal:
snn-test-electrode --snr 5 --duration 30 &
snn-serve --mode electrode

# Or start agent alone and feed it real UDP samples on port 9001
snn-serve --mode electrode
```

**UDP frame format:** `struct.pack('!Hf', 0xABCD, sample_float32)`

## Configuration

All parameters live in a **frozen dataclass** `Config` at
`src/snn_agent/config.py`.  The config is immutable — new instances are
created via factory methods:

```python
from snn_agent.config import Config

cfg = Config()                                    # defaults (trial #65 best)
cfg = Config.from_flat({"l1_n_neurons": 40})      # legacy flat keys (Optuna)
cfg = cfg.with_overrides(sampling_rate_hz=30000)  # dotted overrides
```

### Key parameter groups

| Sub-config | Key fields | Description |
|---|---|---|
| `preprocess` | `enable_bandpass`, `bandpass_lo_hz`, `bandpass_hi_hz`, `decimation_factor` | Front-end filtering |
| `encoder` | `overlap`, `dvm_factor`, `window_depth`, `step_size` | Temporal receptive field shape |
| `dn` | `threshold_factor`, `depression_tau`, `depression_frac` | Attention neuron sensitivity |
| `l1` | `n_neurons`, `dn_weight`, `stdp_ltp`, `stdp_ltd`, `freeze_stdp` | Template layer learning |
| `decoder` | `strategy` (`"rate"` / `"population"` / `"trigger"`), `window_ms`, `threshold` | Control output |
| `lsl` | `stream_name`, `pick_channel`, `bufsize_sec` | LSL connection |
| `synthetic` | `duration_s`, `fs`, `num_units`, `noise_level` | Built-in synthetic recording |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Input                                              │
│  • UDP electrode samples  (electrode mode)          │
│  • LSL stream / .ncs file (lsl mode)                │
│  • SpikeInterface synth   (synthetic mode)          │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  Preprocessor            │  core/preprocessor.py
│  IIR bandpass 300–6 kHz  │
│  Decimation 80→20 kHz    │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  SpikeEncoder            │  core/encoder.py
│  Amplitude bins + shift  │
│  register → bool vector  │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  AttentionNeuron (DN)    │  core/attention.py
│  LIF + synaptic release  │
│  probability (pRel)      │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  TemplateLayer (L1)      │  core/template.py
│  110 LIF neurons, WTA    │
│  snnTorch + custom STDP  │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  ControlDecoder          │  core/decoder.py
│  rate / population /     │
│  trigger → UDP out       │
└──────────────┬───────────┘
          ┌────┴────┐
          ▼         ▼
     UDP out     WebSocket
   (port 9002)  (port 8765)
       │            │
   Experiment    Browser
   hardware     spike raster
               localhost:8080
```

## Project structure

```
snn-agent/
├── pyproject.toml              # Package definition, deps, entry points
├── README.md
├── src/snn_agent/
│   ├── __init__.py             # __version__
│   ├── __main__.py             # python -m snn_agent
│   ├── config.py               # Frozen dataclass Config + sub-configs
│   ├── core/
│   │   ├── preprocessor.py     # Causal bandpass + decimation
│   │   ├── encoder.py          # Temporal RF spike encoder (SpikeEncoder)
│   │   ├── attention.py        # Attention neuron — LIF + pRel (AttentionNeuron)
│   │   ├── template.py         # Template layer — 110 LIF + WTA + STDP (TemplateLayer)
│   │   ├── decoder.py          # Control decoder (ControlDecoder)
│   │   └── pipeline.py         # build_pipeline() / complete_pipeline() factory
│   ├── server/
│   │   ├── app.py              # Async event loop — WebSocket + HTTP + sources
│   │   └── static/index.html   # Live spike raster visualisation
│   ├── eval/
│   │   ├── evaluate.py         # Offline pipeline scorer (SpikeInterface)
│   │   ├── optimize.py         # Optuna hyperparameter search driver
│   │   └── ground_truth.py     # Synthetic ground-truth generation
│   └── io/
│       ├── lsl_player.py       # Replay .ncs files over Lab Streaming Layer
│       └── test_electrode.py   # Synthetic UDP electrode signal generator
├── data/
│   ├── best_config.json        # Best optimisation trial params
│   ├── trials.csv              # Full optimisation history
│   ├── snn-spike-sorting.db    # Optuna SQLite study (resume-safe)
│   └── raw/                    # Raw neural recordings (.ncs)
└── docs/
    ├── annet_architecture.yaml
    ├── optimization_manifest.yaml
    ├── manifesto.json
    ├── scientific_claims.md    # Audited scientific claims
    └── reference/MB2018 - ANNet/
```

## Dependencies

**Core** (installed by default):
```
numpy  scipy  torch  snntorch  websockets  pyyaml
```

**Optional groups** — install via `uv pip install -e ".[group]"`:

| Group | Packages | Purpose |
|---|---|---|
| `lsl` | `mne`, `mne-lsl` | Lab Streaming Layer connectivity |
| `eval` | `spikeinterface`, `optuna` | Offline evaluation & optimisation |
| `all` | All of the above | Full install |
| `dev` | `all` + `pytest`, `ruff` | Development |

## Activation Functions & Dynamics

Mathematical specification of every learnable and non-learnable activation
in the pipeline, derived from the [ANNet architecture](docs/annet_architecture.yaml).

### Temporal Receptive Field Encoding (SpikeEncoder)

**Noise estimation** — Median Absolute Deviation during calibration:

$$\hat{\sigma} = \frac{\mathrm{median}(|x_1|, |x_2|, \dots, |x_N|)}{0.6745}$$

where $N$ = `encoder.noise_init_samples`.  The divisor 0.6745 converts MAD
to a consistent estimator of standard deviation for Gaussian noise.

> **Note:** After calibration, the code tracks $|x_t|$ via an EMA
> ($\alpha$ = `encoder.noise_ema_alpha`) but does **not** re-derive bin
> centres from it.  The initial MAD estimate sets the receptive field
> geometry for the session.

**Amplitude bin half-width:**

$$\Delta v_m = \gamma \cdot \hat{\sigma}$$

where $\gamma$ = `encoder.dvm_factor`. Bin centres are spaced by
$\frac{2 \Delta v_m}{O}$ with $O$ = `encoder.overlap`.

**Afferent activation** — bin $j$ fires when:

$$|c_j - x_t| \leq \Delta v_m$$

The afferent vector is the flattened shift register $\mathbf{a} \in \{0,1\}^{N_c \times D}$
where $N_c$ is the number of centres and $D$ = `encoder.window_depth`.

---

### Attention Neuron (DN) — LIF with Synaptic Depression

**Membrane dynamics** (leaky integrate-and-fire):

$$V_t = V_{t-1} \cdot e^{-1/\tau_m} + \sum_{j \in \mathcal{A}_t} p_{\mathrm{rel},j}(t)$$

where $\mathcal{A}_t$ is the set of active afferents at time $t$ and
$\tau_m$ = `dn.tm_samples`.

**Synaptic release probability** (short-term depression):

$$p_{\mathrm{rel},j}(t) = 1 - \left(1 - p_{\mathrm{rel},j}(t_{\mathrm{last}})\right) e^{-\Delta t / \tau_d}$$

After contributing, each synapse is depressed:

$$p_{\mathrm{rel},j} \leftarrow p_{\mathrm{rel},j} \cdot (1 - f_d)$$

where $\tau_d$ = `dn.depression_tau` and $f_d$ = `dn.depression_frac`.

**Threshold:**

$$\theta_{\mathrm{DN}} = \frac{F_{\mathrm{th}} \cdot O \cdot D}{1 - e^{-1/\tau_m}}$$

where $F_{\mathrm{th}}$ = `dn.threshold_factor`.

**Reset** (set, not additive):

$$V \leftarrow V_{\mathrm{reset}} = F_r \cdot \left(e^{1/\tau_m} - 1\right) \cdot \theta_{\mathrm{DN}}$$

where $F_r$ = `dn.reset_potential_factor`.

---

### Template Layer (L1) — LIF + WTA + STDP

**Input current:**

$$I_t = \mathbf{a}_t^{\top} \mathbf{W} + s_{\mathrm{DN}}(t) \cdot w_{\mathrm{DN}}$$

where $\mathbf{W} \in \mathbb{R}^{N_a \times N}$ are the learnable weights and
$w_{\mathrm{DN}}$ = `l1.dn_weight`.

**LIF membrane** (via snnTorch):

$$V_{i,t} = \beta \, V_{i,t-1} + I_{i,t}, \qquad \beta = e^{-1/\tau_m}$$

where $\tau_m$ = `l1.tm_samples`.

**Spike and reset:**

$$s_{i,t} = \begin{cases} 1 & \text{if } V_{i,t} \geq \theta_{L1} \\ 0 & \text{otherwise} \end{cases}, \qquad V_{i,t} \leftarrow 0 \;\text{ on spike (WTA)}$$

**L1 threshold:**

$$\theta_{L1} = \frac{\left(w_{\mathrm{DN}} + O \cdot (D - k)\right) \left(1 - e^{-S/\tau_m}\right)}{1 - e^{-1/\tau_m}}$$

where $k = 3$ (spike waveform half-width in delay taps) and $S$ = `encoder.step_size`.

**Winner-take-all:** snnTorch's `inhibition=True` — only the neuron with the
highest membrane potential fires per timestep.

**Asymmetric STDP** (applied to the winner $w$):

- *LTD* — global depression at every post-synaptic spike:

$$\Delta W_{j,w}^{-} = \eta^{-} \quad \forall\; j$$

- *LTP* — potentiation for recently active pre-synaptic afferents:

$$\Delta W_{j,w}^{+} = \eta^{+} \quad \text{if } (t - t_j^{\mathrm{pre}}) \leq T_{\mathrm{LTP}}$$

- Weight clip: $W_{j,w} \in [w_{\min},\; w_{\max}]$

where $\eta^{+}$ = `l1.stdp_ltp`, $\eta^{-}$ = `l1.stdp_ltd`,
$T_{\mathrm{LTP}}$ = `l1.stdp_ltp_window`.

---

### Control Decoder

**Rate strategy** — sliding-window weighted sum:

$$u_t = \mathrm{clip}\!\left(\sum_{i=1}^{N} w_i \cdot \frac{n_{i,t}}{T_w},\; -1,\; 1\right)$$

where $n_{i,t}$ counts spikes of neuron $i$ over the window of length $T_w$
(`decoder.window_ms` converted to samples).

**Population strategy** — leaky integrator with reset:

$$\mathcal{I}_t = \mathcal{I}_{t-1} \cdot e^{-1/\tau_c} + \mathbf{w}^{\top} \mathbf{s}_t$$

Emits when $\mathcal{I}_t \geq \theta_c$ (`decoder.threshold`), then resets:

$$\mathcal{I}_t \leftarrow 0 \quad \text{on emission}$$

Output is clipped to $[-1, 1]$.

**Trigger strategy** — binary pulse whenever any L1 neuron fires *and* the DN
is active:

$$u_t = \begin{cases} 1 & \text{if } \exists\, i : s_{i,t}=1 \;\wedge\; s_{\mathrm{DN}}(t)=1 \\ \mathrm{None} & \text{otherwise} \end{cases}$$

**DN confidence** — sliding-window attention activity:

$$\mathrm{conf}_t = \frac{\sum_{k} s_{\mathrm{DN}}(k)}{\min(t,\; T_{\mathrm{dn}})}$$

where $T_{\mathrm{dn}}$ = `decoder.dn_confidence_window_ms` (converted to samples).
During warm-up the denominator equals the number of samples seen so far.

## Hyperparameter Optimisation

Automated search over the 8 most impactful parameters using
[Optuna](https://optuna.org/) (TPE sampler) scored against SpikeInterface
ground-truth synthetic recordings.

### Quick start

```bash
# One-shot evaluation with specific overrides
snn-evaluate --overrides '{"l1_n_neurons": 40}'

# Full optimisation (default trial budget from manifest)
snn-optimize

# Custom budget
snn-optimize --n-trials 80
```

### How it works

1. **`docs/optimization_manifest.yaml`** defines the search space (8 parameters,
   bounds, types) and trial budget.
2. **`snn-evaluate`** generates a synthetic recording via SpikeInterface,
   runs the full pipeline offline (Preprocessor → Encoder → DN → L1 → Decoder),
   and scores L1 clusters against ground truth.
3. **`snn-optimize`** drives an Optuna study: each trial samples a parameter
   set, calls `evaluate_pipeline()`, and reports the accuracy score back.
   TPE learns which regions of the space are promising.

### Outputs

All optimisation artifacts are written to `data/`:

| File | Contents |
|---|---|
| `data/best_config.json` | Best trial's parameters + score |
| `data/trials.csv` | Full history — every trial with params and metrics |
| `data/snn-spike-sorting.db` | SQLite Optuna study (supports resume) |

### Search space

| Parameter | Type | Range | Description |
|---|---|---|---|
| `dn_threshold_factor` | log | [0.15, 0.80] | DN firing threshold scale |
| `l1_dn_weight` | float | [15, 80] | DN excitatory boost to L1 |
| `l1_stdp_ltp` | log | [0.001, 0.02] | LTP amplitude |
| `l1_stdp_ltd` | float | [−0.012, −0.0005] | LTD amplitude |
| `enc_overlap` | int | [4, 20] | Amplitude bin overlap |
| `enc_dvm_factor` | float | [1.0, 3.5] | Noise scale for RF width |
| `enc_step_size` | int | [2, 8] | Temporal subsampling stride |
| `l1_n_neurons` | int | [20, 120] | Template neuron count |

## Multi-terminal workflow

```bash
# Terminal 1 — LSL player (skip if using electrode or synthetic mode)
snn-lsl data/raw/CSC285_0001.ncs

# Terminal 2 — SNN agent
snn-serve

# Terminal 3 — browser
open http://localhost:8080
```

## License

MIT. See `pyproject.toml` for details.
