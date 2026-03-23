"""snn_agent.core — Pipeline components: preprocess → encode → detect → template → decode."""

from snn_agent.core.preprocessor import Preprocessor
from snn_agent.core.encoder import SpikeEncoder
from snn_agent.core.attention import AttentionNeuron
from snn_agent.core.template import TemplateLayer
from snn_agent.core.decoder import ControlDecoder
from snn_agent.core.pipeline import Pipeline, build_pipeline

__all__ = [
    "Preprocessor",
    "SpikeEncoder",
    "AttentionNeuron",
    "TemplateLayer",
    "ControlDecoder",
    "Pipeline",
    "build_pipeline",
]
