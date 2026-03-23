# AGENT-HINT: Core pipeline components. Import order = signal flow.
# preprocess → encode → detect (DN + noise gate) → inhibit → template (L1) → DEC (16 neurons) → decode
"""snn_agent.core — Pipeline components: preprocess → encode → detect → template → DEC → decode."""

from snn_agent.core.preprocessor import Preprocessor
from snn_agent.core.encoder import SpikeEncoder
from snn_agent.core.attention import AttentionNeuron
from snn_agent.core.template import TemplateLayer
from snn_agent.core.decoder import ControlDecoder
from snn_agent.core.inhibition import GlobalInhibitor
from snn_agent.core.noise_gate import NoiseGateNeuron
from snn_agent.core.dec_layer import DECLayer
from snn_agent.core.pipeline import Pipeline, build_pipeline, complete_pipeline

__all__ = [
    "Preprocessor",
    "SpikeEncoder",
    "AttentionNeuron",
    "TemplateLayer",
    "ControlDecoder",
    "GlobalInhibitor",
    "NoiseGateNeuron",
    "DECLayer",
    "Pipeline",
    "build_pipeline",
    "complete_pipeline",
]
