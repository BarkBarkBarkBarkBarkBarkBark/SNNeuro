# AGENT-HINT: Core pipeline components. Import order = signal flow.
# preprocess → encode → detect (DN + noise gate) → inhibit → template (L1) → classify (L2) → decode
"""snn_agent.core — Pipeline components: preprocess → encode → detect → template → decode."""

from snn_agent.core.preprocessor import Preprocessor
from snn_agent.core.encoder import SpikeEncoder
from snn_agent.core.attention import AttentionNeuron
from snn_agent.core.template import TemplateLayer
from snn_agent.core.decoder import ControlDecoder
from snn_agent.core.inhibition import GlobalInhibitor
from snn_agent.core.noise_gate import NoiseGateNeuron
from snn_agent.core.output_layer import ClassificationLayer
from snn_agent.core.pipeline import Pipeline, build_pipeline, complete_pipeline

__all__ = [
    "Preprocessor",
    "SpikeEncoder",
    "AttentionNeuron",
    "TemplateLayer",
    "ControlDecoder",
    "GlobalInhibitor",
    "NoiseGateNeuron",
    "ClassificationLayer",
    "Pipeline",
    "build_pipeline",
    "complete_pipeline",
]
