"""
snn_agent.core.pipeline — Deduplicated pipeline factory.

Builds the complete signal-processing chain from a :class:`Config`.
Eliminates the copy-paste that previously existed in server, evaluate,
and ground_truth modules.
"""

from __future__ import annotations

from typing import NamedTuple

from snn_agent.config import Config
from snn_agent.core.preprocessor import Preprocessor
from snn_agent.core.encoder import SpikeEncoder
from snn_agent.core.attention import AttentionNeuron
from snn_agent.core.template import TemplateLayer
from snn_agent.core.decoder import ControlDecoder
from snn_agent.core.inhibition import GlobalInhibitor
from snn_agent.core.noise_gate import NoiseGateNeuron
from snn_agent.core.output_layer import ClassificationLayer

__all__ = ["Pipeline", "build_pipeline", "complete_pipeline"]


class Pipeline(NamedTuple):
    """All live pipeline components, ready to process samples."""

    preprocessor: Preprocessor
    encoder: SpikeEncoder
    attention: AttentionNeuron
    template: TemplateLayer
    decoder: ControlDecoder
    inhibitor: GlobalInhibitor | None
    noise_gate: NoiseGateNeuron | None
    output_layer: ClassificationLayer | None
    cfg: Config


def build_pipeline(
    cfg: Config,
    *,
    sampling_rate_override: int | None = None,
) -> tuple[Preprocessor, SpikeEncoder, Config]:
    """
    Build the *early* pipeline stages that can be created before
    encoder calibration (Preprocessor + SpikeEncoder).

    After ``encoder.is_calibrated`` becomes True, call
    :func:`complete_pipeline` to create the remaining stages.

    Parameters
    ----------
    cfg : Config
        Full configuration.
    sampling_rate_override : int, optional
        Override sampling rate (e.g. from an LSL stream header).

    Returns
    -------
    preprocessor, encoder, effective_cfg
        The effective_cfg has the correct ``sampling_rate_hz`` after
        decimation applied.
    """
    if sampling_rate_override is not None:
        cfg = cfg.with_overrides(sampling_rate_hz=sampling_rate_override)

    preproc = Preprocessor(cfg)

    # After decimation the downstream components see a lower sample rate
    effective_cfg = cfg.with_overrides(sampling_rate_hz=preproc.effective_fs)
    encoder = SpikeEncoder(effective_cfg)

    return preproc, encoder, effective_cfg


def complete_pipeline(
    cfg: Config,
    effective_cfg: Config,
    preprocessor: Preprocessor,
    encoder: SpikeEncoder,
) -> Pipeline:
    """
    Finish building the pipeline after the encoder has calibrated.

    Parameters
    ----------
    cfg : Config
        Original (pre-decimation) configuration.
    effective_cfg : Config
        Config with ``sampling_rate_hz`` adjusted for decimation.
    preprocessor : Preprocessor
    encoder : SpikeEncoder
        Must be calibrated (``encoder.is_calibrated is True``).

    Returns
    -------
    Pipeline
    """
    n_aff = encoder.n_afferents

    attention = AttentionNeuron(effective_cfg, n_aff)
    template = TemplateLayer(cfg, n_aff)

    # Global post-spike inhibition (default: enabled, 5 ms blanking)
    inhibitor: GlobalInhibitor | None = None
    if cfg.inhibition.enabled:
        inhibitor = GlobalInhibitor(effective_cfg)

    # Kalman noise gate (parallel inhibitory pathway)
    noise_gate_obj: NoiseGateNeuron | None = None
    if cfg.noise_gate.enabled:
        # Use encoder's calibrated noise estimate as baseline
        noise_sigma = encoder.dvm / cfg.encoder.dvm_factor  # reverse: dvm = factor * sigma
        noise_gate_obj = NoiseGateNeuron(effective_cfg, noise_sigma)

    # Optional L2 convergence layer
    output_layer: ClassificationLayer | None = None
    n_decoder_input = cfg.l1.n_neurons
    if cfg.use_l2:
        output_layer = ClassificationLayer(cfg, cfg.l1.n_neurons)
        n_decoder_input = cfg.l2.n_neurons

    decoder = ControlDecoder(effective_cfg, n_decoder_input)

    return Pipeline(
        preprocessor=preprocessor,
        encoder=encoder,
        attention=attention,
        template=template,
        decoder=decoder,
        inhibitor=inhibitor,
        noise_gate=noise_gate_obj,
        output_layer=output_layer,
        cfg=effective_cfg,
    )
