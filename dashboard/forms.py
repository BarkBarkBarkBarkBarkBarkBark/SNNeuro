"""
dashboard.forms — Input configuration form for the SNN pipeline.
"""

from django import forms

SOURCE_CHOICES = [
    ("synthetic", "Synthetic (generated)"),
    ("file", "File (.ncs)"),
    ("electrode", "Electrode (UDP)"),
    ("lsl", "LSL stream"),
]

STRATEGY_CHOICES = [
    ("discrete", "Discrete"),
    ("ttl", "TTL pulse"),
    ("trigger", "Trigger (decay)"),
    ("rate", "Rate"),
    ("population", "Population"),
]


class InputConfigForm(forms.Form):
    # Source
    source_type = forms.ChoiceField(
        choices=SOURCE_CHOICES,
        initial="synthetic",
        widget=forms.RadioSelect,
        label="Input source",
    )
    num_channels = forms.IntegerField(
        min_value=1,
        max_value=32,
        initial=1,
        label="Number of channels",
        help_text="How many parallel recording channels to sort simultaneously.",
    )

    # Synthetic params
    synth_duration_s = forms.FloatField(
        min_value=1.0,
        max_value=600.0,
        initial=20.0,
        label="Duration (s)",
        required=False,
    )
    synth_num_units = forms.IntegerField(
        min_value=1,
        max_value=20,
        initial=2,
        label="Number of units",
        required=False,
    )
    synth_noise_level = forms.FloatField(
        min_value=1.0,
        max_value=50.0,
        initial=8.0,
        label="Noise level",
        required=False,
    )
    synth_seed = forms.IntegerField(
        min_value=0,
        initial=42,
        label="Random seed",
        required=False,
    )

    # File params
    file_path = forms.CharField(
        max_length=512,
        required=False,
        label="Recording file (.ncs)",
        widget=forms.TextInput(attrs={"placeholder": "/path/to/recording.ncs", "spellcheck": "false"}),
    )

    # Signal params (common)
    decoder_strategy = forms.ChoiceField(
        choices=STRATEGY_CHOICES,
        initial="discrete",
        label="Decoder strategy",
    )
