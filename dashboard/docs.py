"""
dashboard.docs — Documentation browser views.

Serves markdown files from the project ``docs/`` folder, rendered to HTML
on-the-fly.  Also exposes a JSON endpoint with the current hyperparameters
(defaults + best_config.json overrides) so the Docs tab can show live values.
"""

from __future__ import annotations

import json
from dataclasses import fields, asdict
from pathlib import Path

import markdown
from django.http import JsonResponse, Http404
from django.shortcuts import render, redirect
from django.views.decorators.http import require_GET

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _PROJECT_ROOT / "docs"
_BEST_CONFIG = _PROJECT_ROOT / "data" / "best_config.json"

# Markdown renderer with useful extensions
_MD = markdown.Markdown(
    extensions=[
        "tables",
        "fenced_code",
        "codehilite",
        "toc",
        "attr_list",
        "md_in_html",
    ],
    extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": False},
        "toc": {"permalink": True, "toc_depth": "2-4"},
    },
)

# File extensions we'll render / serve
_BROWSABLE = {".md", ".yaml", ".yml", ".json"}

# Friendly display names (falls back to title-cased filename)
_DISPLAY_NAMES: dict[str, str] = {
    "neuron_guide.md": "Neuron & Component Guide",
    "pipeline_study.md": "Pipeline Study",
    "scientific_principles.md": "Scientific Principles",
    "scientific_claims.md": "Scientific Claims",
    "optimization_guide.md": "Optimization Guide",
    "optimization_manifest.yaml": "Optimization Manifest",
    "optimization_rules.md": "Optimization Rules",
    "django_migration_roadmap.md": "Django Migration Roadmap",
    "annet_architecture.yaml": "ANNet Architecture",
    "manifesto.json": "Project Manifesto",
}


def _list_docs() -> list[dict]:
    """Return a sorted list of {name, slug, ext} dicts for browsable docs."""
    docs = []
    if not _DOCS_DIR.is_dir():
        return docs
    for f in sorted(_DOCS_DIR.iterdir()):
        if f.is_file() and f.suffix in _BROWSABLE:
            slug = f.stem
            docs.append({
                "name": _DISPLAY_NAMES.get(f.name, f.stem.replace("_", " ").title()),
                "slug": slug,
                "ext": f.suffix,
                "filename": f.name,
            })
    return docs


def _render_file(path: Path) -> str:
    """Read a file and return HTML content."""
    raw = path.read_text(encoding="utf-8")

    if path.suffix == ".md":
        _MD.reset()
        return _MD.convert(raw)

    # YAML / JSON — wrap in a fenced code block, then render
    lang = "yaml" if path.suffix in (".yaml", ".yml") else "json"
    wrapped = f"```{lang}\n{raw}\n```"
    _MD.reset()
    return _MD.convert(wrapped)


# ── Views ────────────────────────────────────────────────────────────────────

@require_GET
def docs_index(request):
    """Redirect bare /docs/ to the first available document."""
    docs = _list_docs()
    if docs:
        return redirect("docs_page", slug=docs[0]["slug"])
    raise Http404("No documentation files found in docs/")


@require_GET
def docs_page(request, slug: str):
    """Render a single documentation page with sidebar navigation."""
    docs = _list_docs()

    # Find the requested doc
    target = None
    for d in docs:
        if d["slug"] == slug:
            target = d
            break

    if target is None:
        raise Http404(f"Document '{slug}' not found")

    filepath = _DOCS_DIR / target["filename"]
    if not filepath.is_file():
        raise Http404(f"File not found: {target['filename']}")

    content_html = _render_file(filepath)

    return render(request, "dashboard/docs.html", {
        "docs": docs,
        "active_slug": slug,
        "active_name": target["name"],
        "content_html": content_html,
    })


# ── JSON API: current hyperparameters ────────────────────────────────────────

@require_GET
def api_config(request) -> JsonResponse:
    """
    GET /docs/api/config/

    Returns a JSON object with two keys:
      - ``defaults``: all Config dataclass defaults (grouped by sub-config)
      - ``best``: overrides from data/best_config.json (if it exists)
    """
    from snn_agent.config import Config

    cfg = Config()
    defaults = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if hasattr(val, "__dataclass_fields__"):
            defaults[f.name] = asdict(val)
        else:
            defaults[f.name] = val

    best = {}
    if _BEST_CONFIG.is_file():
        try:
            raw = json.loads(_BEST_CONFIG.read_text())
            best = raw.get("parameters", raw)
        except Exception:
            pass

    return JsonResponse({"defaults": defaults, "best": best})
