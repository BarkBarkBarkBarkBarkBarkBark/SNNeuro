# AGENT-HINT: Genetic mutation optimizer — breeds top trials via crossover + mutation.
# PURPOSE: Takes top-K trials from an Optuna study (or trials.csv), groups params
#          into functional "gene blocks", and creates offspring by recombining
#          blocks from different parents + small Gaussian mutations.
# OUTPUT:  Same format as optimize.py — best_config.json + genetic_trials.csv in data/
# SEE ALSO: optimize.py (TPE search), evaluate.py (objective), config.py,
#           docs/optimization_manifest.yaml (search space + bounds)
"""
snn_agent.eval.genetic — Genetic crossover + mutation optimizer.

Takes the top-performing trials from a previous Optuna run and breeds
new configurations by:

1. **Selection** — rank trials by accuracy, keep top-K as the elite pool.
2. **Crossover** — pick two parents, swap entire "gene blocks" (functionally
   coupled parameter groups) to produce an offspring.
3. **Mutation** — apply small Gaussian perturbations to each parameter,
   respecting type (int/float/log) and manifest bounds.
4. **Evaluation** — score each offspring via ``evaluate_pipeline()``.
5. **Elitism** — the best parent always survives into the next generation.

Gene blocks reflect the pipeline architecture — parameters that interact
physically are inherited together so good sub-configurations survive::

    ENC  = {enc_overlap, enc_dvm_factor, enc_step_size}
    DN   = {dn_threshold_factor, l1_dn_weight, dn_depression_tau, dn_depression_frac}
    STDP = {l1_stdp_ltp, l1_stdp_ltd, l1_n_neurons}
    INH  = {inh_duration_ms, inh_strength_threshold}
    NG   = {ng_process_noise, ng_inhibit_below_sd, ng_suppression_factor, ng_ema_alpha}
    DEC  = {dec_dn_window_ms}
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from snn_agent.eval.evaluate import multi_evaluate

__all__ = ["run_genetic"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data"

# ── Gene blocks: functionally coupled parameter groups ────────────────────────
GENE_BLOCKS: dict[str, list[str]] = {
    "ENC":  ["enc_overlap", "enc_dvm_factor", "enc_step_size"],
    "DN":   ["dn_threshold_factor", "l1_dn_weight", "dn_depression_tau", "dn_depression_frac"],
    "STDP": ["l1_stdp_ltp", "l1_stdp_ltd", "l1_n_neurons"],
    "INH":  ["inh_duration_ms", "inh_strength_threshold"],
    "NG":   ["ng_process_noise", "ng_inhibit_below_sd", "ng_suppression_factor", "ng_ema_alpha"],
    "DEC":  ["dec_dn_window_ms"],
}


def _find_manifest() -> Path:
    for candidate in [ROOT / "docs" / "optimization_manifest.yaml",
                      ROOT / "optimization_manifest.yaml"]:
        if candidate.exists():
            return candidate
    return ROOT / "docs" / "optimization_manifest.yaml"


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Load elite pool from trials.csv ──────────────────────────────────────────
def load_elite(
    trials_path: Path,
    top_k: int = 10,
    min_score: float = 0.01,
) -> list[dict[str, Any]]:
    """Load top-K trials from trials.csv, sorted by value descending.

    Each returned dict has ``"params"`` (flat param dict) and ``"score"``.
    Trials with score < min_score are excluded (dead configs).
    """
    rows: list[dict] = []
    with open(trials_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["value"])
            if score < min_score:
                continue
            params = {}
            for k, v in row.items():
                if k in ("trial", "value", "accuracy", "n_active",
                          "precision", "recall", "runtime_s", "total_spikes"):
                    continue
                if v == "":
                    continue
                # Coerce to proper type
                try:
                    if "." in v or "e" in v.lower():
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except (ValueError, TypeError):
                    params[k] = v
            rows.append({"params": params, "score": score})

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


# ── Crossover: block-level recombination ─────────────────────────────────────
def crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    rng: np.random.Generator,
    crossover_rate: float = 0.5,
) -> dict[str, Any]:
    """Create offspring by swapping entire gene blocks between two parents.

    For each block, flip a coin: take the block from parent A or parent B.
    Parameters not covered by any block are taken from parent A.
    """
    child: dict[str, Any] = dict(parent_a)  # start with A as base

    for block_name, block_params in GENE_BLOCKS.items():
        if rng.random() < crossover_rate:
            # Take this block from parent B
            for p in block_params:
                if p in parent_b:
                    child[p] = parent_b[p]

    return child


# ── Mutation: bounded Gaussian perturbation ──────────────────────────────────
def mutate(
    params: dict[str, Any],
    param_defs: dict,
    rng: np.random.Generator,
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.15,
) -> dict[str, Any]:
    """Apply small mutations respecting type and manifest bounds.

    Each parameter has ``mutation_rate`` probability of being mutated.
    Mutation is a Gaussian perturbation scaled to ``mutation_strength``
    fraction of the parameter's range, clamped to manifest bounds.

    - ``float`` params: additive Gaussian in linear space.
    - ``log`` params: additive Gaussian in log space (multiplicative).
    - ``int`` params: additive Gaussian rounded to nearest int (or step).
    - ``bool`` params: flip with probability mutation_rate.
    """
    mutated = dict(params)

    for name, spec in param_defs.items():
        if name not in mutated:
            continue
        if rng.random() > mutation_rate:
            continue

        ptype = spec["type"]
        lo, hi = spec.get("low", 0), spec.get("high", 1)
        val = mutated[name]

        if ptype == "bool":
            mutated[name] = not val
            continue

        if ptype == "log":
            # Mutate in log space
            log_lo, log_hi = math.log(lo), math.log(hi)
            log_val = math.log(max(val, 1e-15))
            sigma = mutation_strength * (log_hi - log_lo)
            log_new = log_val + rng.normal(0, sigma)
            log_new = float(np.clip(log_new, log_lo, log_hi))
            mutated[name] = math.exp(log_new)
        elif ptype == "int":
            step = spec.get("step", 1)
            span = hi - lo
            sigma = mutation_strength * span
            raw = val + rng.normal(0, sigma)
            # Round to nearest step
            rounded = int(round((raw - lo) / step)) * step + int(lo)
            mutated[name] = int(np.clip(rounded, int(lo), int(hi)))
        else:  # float
            span = hi - lo
            sigma = mutation_strength * span
            new_val = val + rng.normal(0, sigma)
            mutated[name] = float(np.clip(new_val, lo, hi))

    return mutated


# ── Evaluate one offspring ───────────────────────────────────────────────────
def _eval_one(
    params: dict[str, Any],
    metric: str,
    trial_id: int,
    scenarios: list[dict] | None = None,
    score_after_s: float | None = 15.0,
    delta_time: float = 2.0,
) -> dict[str, Any]:
    """Evaluate a single parameter config across multiple scenarios."""
    print(f"  ▸ Trial {trial_id:3d} started …", end="", flush=True)
    t0 = time.perf_counter()

    try:
        result = multi_evaluate(
            cfg_overrides=params,
            scenarios=scenarios,
            score_after_s=score_after_s,
            delta_time=delta_time,
            verbose=False,
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  FAILED ({dt:.0f}s) — {e}")
        return {
            "params": params, "score": 0.0,
            "accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f_half": 0.0,
            "n_active": 0, "total_spikes": 0, "runtime_s": dt,
        }

    score = result.get(metric, 0.0)
    dt = time.perf_counter() - t0

    print(
        f"  f½={score:.4f}  acc={result.get('accuracy',0):.4f}  "
        f"prec={result.get('precision',0):.4f}  "
        f"rec={result.get('recall',0):.4f}  ({dt:.0f}s)"
    )

    return {
        "params": params,
        "score": float(score),
        "accuracy": result.get("accuracy", 0.0),
        "recall": result.get("recall", 0.0),
        "precision": result.get("precision", 0.0),
        "f_half": result.get("f_half", 0.0),
        "n_active": result.get("n_active", 0),
        "total_spikes": result.get("total_spikes", 0),
        "runtime_s": dt,
    }


# ── Main genetic loop ────────────────────────────────────────────────────────
def run_genetic(
    manifest_path: Path | None = None,
    trials_path: Path | None = None,
    top_k: int = 10,
    n_offspring: int = 160,
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.15,
    crossover_rate: float = 0.5,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run the genetic crossover + mutation optimizer.

    Parameters
    ----------
    manifest_path : Path
        Search space definition (for bounds and types).
    trials_path : Path
        CSV from a previous Optuna run (default: data/trials.csv).
    top_k : int
        Number of elite parents to select.
    n_offspring : int
        Total offspring to evaluate.
    mutation_rate : float
        Per-parameter probability of mutation (0–1).
    mutation_strength : float
        Gaussian sigma as fraction of param range (0–1).
    crossover_rate : float
        Per-block probability of swapping from parent B (0–1).
    output_dir : Path
        Where to write results.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with best params and score.
    """
    if manifest_path is None:
        manifest_path = _find_manifest()
    if trials_path is None:
        trials_path = DATA_DIR / "trials.csv"
    if output_dir is None:
        output_dir = DATA_DIR

    manifest = _load_manifest(manifest_path)
    param_defs = manifest["parameters"]
    obj_cfg = manifest.get("objective", {})
    metric = obj_cfg.get("metric", "f_half")
    scenarios = obj_cfg.get("scenarios", None)
    score_after_s = obj_cfg.get("score_after_s", 15.0)
    delta_time_val = obj_cfg.get("delta_time", 2.0)

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load elite pool ───────────────────────────────────────
    elite = load_elite(trials_path, top_k=top_k)
    if len(elite) < 2:
        raise ValueError(
            f"Need at least 2 viable trials in {trials_path}, "
            f"found {len(elite)} with score > 0.01"
        )

    print("=" * 60)
    print("SNN Agent — Genetic Crossover Optimizer")
    print("=" * 60)
    print(f"  Elite pool   : {len(elite)} parents (top-{top_k})")
    print(f"  Best parent  : acc={elite[0]['score']:.4f}")
    print(f"  Worst parent : acc={elite[-1]['score']:.4f}")
    print(f"  Offspring    : {n_offspring}")
    print(f"  Mutation     : rate={mutation_rate}, strength={mutation_strength}")
    print(f"  Crossover    : rate={crossover_rate} (block-level)")
    print(f"  Gene blocks  : {', '.join(GENE_BLOCKS.keys())}")
    print(f"  Metric       : {metric}")
    print()

    # Show elite parents
    print("  ── Elite Parents ──")
    for i, e in enumerate(elite):
        print(f"     #{i:2d}  acc={e['score']:.4f}")
    print()

    # ── Step 2: Generate offspring via crossover + mutation ────────────
    all_results: list[dict] = []

    # Always re-evaluate the best parent first (elitism, trial 0)
    print("  ── Generation 0: Re-evaluate best parent ──")
    best_parent_result = _eval_one(
        dict(elite[0]["params"]), metric, trial_id=0,
        scenarios=scenarios, score_after_s=score_after_s, delta_time=delta_time_val,
    )
    all_results.append(best_parent_result)

    print(f"\n  ── Generations 1–{n_offspring - 1}: Crossover + Mutation ──")

    # Fitness-proportional selection weights (softmax of scores)
    scores = np.array([e["score"] for e in elite])
    # Temperature-scaled softmax to favour better parents without ignoring others
    temperature = 0.1
    logits = scores / max(temperature, 1e-10)
    logits -= logits.max()  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum()

    for trial_id in range(1, n_offspring):
        # Pick two parents (fitness-proportional, no replacement)
        if len(elite) >= 2:
            idx = rng.choice(len(elite), size=2, replace=False, p=weights)
        else:
            idx = [0, 0]
        parent_a = elite[idx[0]]["params"]
        parent_b = elite[idx[1]]["params"]

        # Crossover
        child = crossover(parent_a, parent_b, rng, crossover_rate)

        # Mutation
        child = mutate(child, param_defs, rng, mutation_rate, mutation_strength)

        # Evaluate
        result = _eval_one(
            child, metric, trial_id=trial_id,
            scenarios=scenarios, score_after_s=score_after_s, delta_time=delta_time_val,
        )
        all_results.append(result)

        # Progress update every 20 trials
        if (trial_id + 1) % 20 == 0:
            running_best = max(all_results, key=lambda r: r["score"])
            print(
                f"\n  ── Progress: {trial_id + 1}/{n_offspring} done — "
                f"best so far: f½={running_best['score']:.4f} ──\n"
            )

    # ── Step 3: Find best and export ──────────────────────────────────
    all_results.sort(key=lambda r: r["score"], reverse=True)
    best = all_results[0]

    print()
    print("=" * 60)
    print(f"  Best trial   : f½={best['score']:.4f}")
    print(f"  Accuracy     : {best['accuracy']:.4f}")
    print(f"  Recall       : {best['recall']:.4f}")
    print(f"  Precision    : {best['precision']:.4f}")
    print(f"  Active L1    : {best['n_active']}")
    print(f"  Total spikes : {best['total_spikes']}")
    for k, v in sorted(best["params"].items()):
        vfmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"    {k:30s} = {vfmt}")
    print("=" * 60)

    # Export best_config.json (same format as optimize.py)
    best_config_path = output_dir / "best_config.json"
    payload = {
        "study_name": "snn-genetic",
        "best_trial": 0,
        "best_score": best["score"],
        "metric": metric,
        "parameters": best["params"],
        "user_attrs": {
            "accuracy": best["accuracy"],
            "f_half": best.get("f_half", 0.0),
            "n_active": best["n_active"],
            "precision": best["precision"],
            "recall": best["recall"],
            "runtime_s": best["runtime_s"],
            "total_spikes": best["total_spikes"],
        },
    }
    with open(best_config_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  📄 Best config written to {best_config_path}")

    # Export trials CSV (same format as optimize.py)
    trials_out = output_dir / "genetic_trials.csv"
    param_names = sorted(best["params"].keys())
    attr_names = ["accuracy", "f_half", "n_active", "precision", "recall", "runtime_s", "total_spikes"]
    header = ["trial", "value"] + param_names + attr_names

    with open(trials_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, r in enumerate(all_results):
            row = [i, r["score"]]
            row += [r["params"].get(p, "") for p in param_names]
            row += [r.get(a, "") for a in attr_names]
            writer.writerow(row)
    print(f"  📊 {len(all_results)} trials written to {trials_out}")

    return best


# ── CLI entry point ──────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point (``snn-genetic``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SNN Genetic Crossover Optimizer — breed top trials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=str(_find_manifest()),
        help="Path to optimization_manifest.yaml",
    )
    parser.add_argument(
        "--trials", type=str, default=str(DATA_DIR / "trials.csv"),
        help="Path to trials.csv from a previous Optuna run",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of elite parents to select",
    )
    parser.add_argument(
        "--n-offspring", type=int, default=160,
        help="Total offspring to evaluate",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.3,
        help="Per-parameter probability of mutation",
    )
    parser.add_argument(
        "--mutation-strength", type=float, default=0.15,
        help="Gaussian sigma as fraction of param range",
    )
    parser.add_argument(
        "--crossover-rate", type=float, default=0.5,
        help="Per-block probability of swapping from second parent",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_DIR),
        help="Directory for output files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility",
    )
    args = parser.parse_args()

    run_genetic(
        manifest_path=Path(args.manifest),
        trials_path=Path(args.trials),
        top_k=args.top_k,
        n_offspring=args.n_offspring,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
