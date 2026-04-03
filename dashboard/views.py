"""
dashboard.views — Page views for the SNN web dashboard.

Two pages:
  /          → input configuration (InputConfigForm)
  /monitor/  → live visualisation (no form — reads session config)
"""

from __future__ import annotations

from django.urls import reverse
from django.shortcuts import render, redirect
from .forms import InputConfigForm


def input_page(request):
    """Input configuration page: configure source, channels, synthetic params."""
    if request.method == "POST":
        form = InputConfigForm(request.POST)
        if form.is_valid():
            d = form.cleaned_data
            # Pass key params to monitor via URL query string — no session needed.
            url = reverse("monitor") + (
                f"?channels={d.get('num_channels', 1)}"
                f"&source={d.get('source_type', 'synthetic')}"
            )
            return redirect(url)
    else:
        form = InputConfigForm()

    return render(request, "dashboard/input.html", {"form": form})


def monitor_page(request):
    """Live monitor page: shows all canvases, controls, network topology."""
    context = {
        "num_channels": int(request.GET.get("channels", 1)),
        "source_type": request.GET.get("source", "synthetic"),
    }
    return render(request, "dashboard/monitor.html", context)
