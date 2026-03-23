"""
snn_agent.config — Single-source-of-truth configuration as a frozen dataclass.

All parameters for the SNN pipeline live here.  Override individual fields via
``Config.from_dict({...})`` or ``Config.with_overrides({...})``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-configs (nested, immutable)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    enable_bandpass: bool = True
    bandpass_lo_hz: int = 300
    bandpass_hi_hz: int = 6000
    bandpass_order: int = 2
    enable_decimation: bool = True
    decimation_factor: int = 4


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    overlap: int = 9
    dvm_factor: float = 2.055
    step_size: int = 2
    window_depth: int = 10
    noise_init_samples: int = 8000


@dataclass(frozen=True, slots=True)
class DNConfig:
    """Attention neuron (detection neuron) parameters."""
    tm_samples: int = 2
    threshold_factor: float = 0.3825
    reset_potential_factor: float = 0.15
    depression_tau: int = 400
    depression_frac: float = 0.01613


@dataclass(frozen=True, slots=True)
class L1Config:
    """Template layer parameters."""
    n_neurons: int = 110
    tm_samples: int = 2
    refractory_samples: int = 1
    dn_weight: float = 113.26
    init_w_min: float = 0.4
    init_w_max: float = 1.0
    w_lo: float = 0.0
    w_hi: float = 1.0
    stdp_ltp: float = 0.016
    stdp_ltp_window: int = 4
    stdp_ltd: float = -0.00763
    freeze_stdp: bool = False


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    strategy: Literal["rate", "population", "trigger"] = "rate"
    window_ms: float = 5.0
    weights: list[float] | None = None
    threshold: float = 0.5
    leaky_tau_ms: float = 10.0
    dn_confidence_window_ms: float = 5.0


@dataclass(frozen=True, slots=True)
class LSLConfig:
    stream_name: str = "NCS-Replay"
    pick_channel: str | None = None
    bufsize_sec: float = 5.0
    poll_interval_s: float = 0.0005


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    duration_s: float = 20.0
    fs: int = 30_000
    num_units: int = 2
    noise_level: float = 8.0
    seed: int = 42
    realtime: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Config:
    """
    Immutable project configuration.

    Create the default via ``Config()``.
    Override fields via ``Config.with_overrides({"l1.n_neurons": 40})``
    or the flat-key form ``Config.from_flat({"l1_n_neurons": 40})``.
    """

    mode: Literal["electrode", "lsl", "synthetic"] = "lsl"

    # Network ports
    ws_port: int = 8765
    http_port: int = 8080
    udp_electrode_port: int = 9001
    udp_control_port: int = 9002
    control_target_host: str = "127.0.0.1"

    # Signal
    sampling_rate_hz: int = 80_000

    # Sub-configs
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dn: DNConfig = field(default_factory=DNConfig)
    l1: L1Config = field(default_factory=L1Config)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    lsl: LSLConfig = field(default_factory=LSLConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    # Viz
    broadcast_every: int = 5

    # ── Convenience factories ─────────────────────────────────────────
    def to_dict_flat(self) -> dict:
        """Serialize to a flat dict using legacy key names."""
        flat: dict = {}
        for flat_key, (group, attr) in _FLAT_MAP.items():
            if group is None:
                flat[flat_key] = getattr(self, attr, None)
            else:
                sub = getattr(self, group, None)
                if sub is not None:
                    flat[flat_key] = getattr(sub, attr, None)
        return flat

    def with_overrides(self, **kw) -> Config:
        """Return a new Config with top-level fields replaced."""
        return _replace_recursive(self, kw)

    @classmethod
    def from_flat(cls, flat: dict) -> Config:
        """
        Build a Config from a flat dict using the legacy key convention.

        Maps old flat keys like ``"l1_n_neurons"`` → ``L1Config(n_neurons=...)``.
        Unknown keys are silently ignored so Optuna trial dicts just work.
        """
        return _from_flat(flat)

    def effective_fs(self) -> int:
        """Sampling rate after optional decimation."""
        if self.preprocess.enable_decimation:
            return self.sampling_rate_hz // self.preprocess.decimation_factor
        return self.sampling_rate_hz


# ─────────────────────────────────────────────────────────────────────────────
#  Flat-key mapping  (legacy CFG dict → structured Config)
# ─────────────────────────────────────────────────────────────────────────────
_FLAT_MAP: dict[str, tuple[str | None, str]] = {
    # top-level
    "mode": (None, "mode"),
    "ws_port": (None, "ws_port"),
    "http_port": (None, "http_port"),
    "udp_electrode_port": (None, "udp_electrode_port"),
    "udp_control_port": (None, "udp_control_port"),
    "control_target_host": (None, "control_target_host"),
    "sampling_rate_hz": (None, "sampling_rate_hz"),
    "broadcast_every": (None, "broadcast_every"),
    # preprocess
    "enable_bandpass": ("preprocess", "enable_bandpass"),
    "bandpass_lo_hz": ("preprocess", "bandpass_lo_hz"),
    "bandpass_hi_hz": ("preprocess", "bandpass_hi_hz"),
    "bandpass_order": ("preprocess", "bandpass_order"),
    "enable_decimation": ("preprocess", "enable_decimation"),
    "decimation_factor": ("preprocess", "decimation_factor"),
    # encoder
    "enc_overlap": ("encoder", "overlap"),
    "enc_dvm_factor": ("encoder", "dvm_factor"),
    "enc_step_size": ("encoder", "step_size"),
    "enc_window_depth": ("encoder", "window_depth"),
    "enc_noise_init_samples": ("encoder", "noise_init_samples"),
    # dn
    "dn_tm_samples": ("dn", "tm_samples"),
    "dn_threshold_factor": ("dn", "threshold_factor"),
    "dn_reset_potential_factor": ("dn", "reset_potential_factor"),
    "dn_depression_tau": ("dn", "depression_tau"),
    "dn_depression_frac": ("dn", "depression_frac"),
    # l1
    "l1_n_neurons": ("l1", "n_neurons"),
    "l1_tm_samples": ("l1", "tm_samples"),
    "l1_refractory_samples": ("l1", "refractory_samples"),
    "l1_dn_weight": ("l1", "dn_weight"),
    "l1_init_w_min": ("l1", "init_w_min"),
    "l1_init_w_max": ("l1", "init_w_max"),
    "l1_stdp_ltp": ("l1", "stdp_ltp"),
    "l1_stdp_ltd": ("l1", "stdp_ltd"),
    "l1_stdp_ltp_window": ("l1", "stdp_ltp_window"),
    "l1_freeze_stdp": ("l1", "freeze_stdp"),
    # decoder
    "ctrl_strategy": ("decoder", "strategy"),
    "ctrl_window_ms": ("decoder", "window_ms"),
    "ctrl_weights": ("decoder", "weights"),
    "ctrl_threshold": ("decoder", "threshold"),
    "ctrl_leaky_tau_ms": ("decoder", "leaky_tau_ms"),
    "ctrl_dn_confidence_window_ms": ("decoder", "dn_confidence_window_ms"),
    # lsl
    "lsl_stream_name": ("lsl", "stream_name"),
    "lsl_pick_channel": ("lsl", "pick_channel"),
    "lsl_bufsize_sec": ("lsl", "bufsize_sec"),
    "lsl_poll_interval_s": ("lsl", "poll_interval_s"),
    # synthetic
    "synth_duration_s": ("synthetic", "duration_s"),
    "synth_fs": ("synthetic", "fs"),
    "synth_num_units": ("synthetic", "num_units"),
    "synth_noise_level": ("synthetic", "noise_level"),
    "synth_seed": ("synthetic", "seed"),
    "synth_realtime": ("synthetic", "realtime"),
}


def _from_flat(flat: dict) -> Config:
    """Convert legacy flat-key dict to a structured Config."""
    top: dict = {}
    subs: dict[str, dict] = {}

    for flat_key, value in flat.items():
        mapping = _FLAT_MAP.get(flat_key)
        if mapping is None:
            continue  # skip unknown keys
        group, attr = mapping
        if group is None:
            top[attr] = value
        else:
            subs.setdefault(group, {})[attr] = value

    # Build sub-configs
    if "preprocess" in subs:
        top["preprocess"] = PreprocessConfig(**subs["preprocess"])
    if "encoder" in subs:
        top["encoder"] = EncoderConfig(**subs["encoder"])
    if "dn" in subs:
        top["dn"] = DNConfig(**subs["dn"])
    if "l1" in subs:
        top["l1"] = L1Config(**subs["l1"])
    if "decoder" in subs:
        top["decoder"] = DecoderConfig(**subs["decoder"])
    if "lsl" in subs:
        top["lsl"] = LSLConfig(**subs["lsl"])
    if "synthetic" in subs:
        top["synthetic"] = SyntheticConfig(**subs["synthetic"])

    return Config(**top)


def _replace_recursive(cfg: Config, kw: dict) -> Config:
    """Return a new Config with fields overridden (supports nested dataclass fields)."""
    from dataclasses import replace as _replace
    clean: dict = {}
    for k, v in kw.items():
        current = getattr(cfg, k, None)
        if isinstance(v, dict) and current is not None and hasattr(current, "__dataclass_fields__"):
            clean[k] = _replace(current, **v)  # type: ignore[type-var]
        else:
            clean[k] = v
    return _replace(cfg, **clean)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level default (importable convenience)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Config()
