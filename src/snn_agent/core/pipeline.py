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

__all__ = ["Pipeline", "build_pipeline"]


class Pipeline(NamedTuple):
    """All live pipeline components, ready to process samples."""

    preprocessor: Preprocessor
    encoder: SpikeEncoder
    attention: AttentionNeuron
    template: TemplateLayer
    decoder: ControlDecoder
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
    decoder = ControlDecoder(effective_cfg, cfg.l1.n_neurons)

    return Pipeline(
        preprocessor=preprocessor,
        encoder=encoder,
        attention=attention,
        template=template,
        decoder=decoder,
        cfg=effective_cfg,
    )
