"""Model Strategy Profiles — Best Known Configurations (BKC) for specific local models.

Each profile captures the optimal cascade settings for a model family, including:
- LLM reviewer strategy (on/off, bypass on pass)
- Thinking mode per agent role
- Technique flags (which repair/enhancement techniques help this model)
- Sampling strategy (best-of-N)

Profiles can be hand-tuned or auto-generated from ablation benchmark results
via `scripts/build_profiles.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict


@dataclass
class TechniqueScore:
    """Measured contribution of a single technique for a specific model."""
    technique: str
    delta_probes: float = 0.0    # baseline - no_technique (positive = technique helps)
    delta_tokens: float = 0.0    # token cost difference
    delta_time: float = 0.0      # wall time difference (seconds)
    runs: int = 0                # number of benchmark runs backing this score
    enabled: bool = True         # whether this technique should be on for this model


@dataclass
class StrategyProfile:
    name: str
    # Reviewer strategy
    use_llm_reviewer: bool = True
    reviewer_bypass_on_pass: bool = True
    # Thinking mode
    think_architect: bool = False
    think_coder: bool = False
    # Sampling
    best_of_n: int = 1
    # Technique flags — all default True, disabled per model when benchmarks show harm
    import_fixes: bool = True
    ast_fixes: bool = True
    test_calibration: bool = True
    micro_fix: bool = True
    lesson_injection: bool = True
    spec_literals: bool = True
    test_triage: bool = True         # Per-failure investigation cascade
    runtime_fixes: bool = False      # Benchmarked as neutral; off by default
    defensive_guards: bool = False   # Benchmarked as harmful; off by default
    # Benchmark evidence (populated by build_profiles.py)
    technique_scores: list[TechniqueScore] = field(default_factory=list)
    baseline_probes: float = 0.0     # average probes on baseline combo
    benchmark_runs: int = 0          # total ablation runs backing this profile


# ── Hand-tuned profiles (overridden by build_profiles.py when data exists) ──

# Model family defaults — used as starting point for models without benchmark data.
# Keyed by the shortest unique model identifier that fuzzy-matches Ollama tags.

STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    # ── Qwen 2.5 Coder family ──
    "qwen2.5-coder:7b": StrategyProfile(
        name="qwen2.5-coder:7b",
        use_llm_reviewer=True,
        best_of_n=1,
        baseline_probes=69.5,
        benchmark_runs=4,
    ),
    "qwen2.5-coder:14b": StrategyProfile(
        name="qwen2.5-coder:14b",
        use_llm_reviewer=True,
        best_of_n=2,
        baseline_probes=70.0,
        benchmark_runs=2,
    ),
    "qwen2.5-coder:32b": StrategyProfile(
        name="qwen2.5-coder:32b",
        use_llm_reviewer=True,
        best_of_n=1,  # 32B is consistent enough
    ),

    # ── Qwen 3 family ──
    "qwen3:8b": StrategyProfile(
        name="qwen3:8b",
        use_llm_reviewer=False,      # reasoning model, pedantic reviewer
        think_architect=True,
        reviewer_bypass_on_pass=True,
    ),
    "qwen3:30b": StrategyProfile(
        name="qwen3:30b",
        use_llm_reviewer=True,       # 30B is reliable
        think_architect=True,
        best_of_n=1,
    ),

    # ── Qwen 3.5 Coder family ──
    "qwen3.5:9b": StrategyProfile(
        name="qwen3.5:9b",
        use_llm_reviewer=False,      # pedantic reviewer; trust deterministic gates
        think_architect=True,
        think_coder=False,
        baseline_probes=76.0,
        benchmark_runs=2,
    ),
    "qwen3.5-coder": StrategyProfile(
        name="qwen3.5-coder",
        use_llm_reviewer=False,      # MANDATORY: stop self-sabotage
        think_architect=True,
        think_coder=False,
        baseline_probes=76.0,
        benchmark_runs=2,
    ),

    # ── DeepSeek Coder family ──
    "deepseek-coder-v2:16b": StrategyProfile(
        name="deepseek-coder-v2:16b",
        use_llm_reviewer=True,
        spec_literals=True,          # critical: prevents naming divergence
        baseline_probes=66.0,
        benchmark_runs=2,
    ),
    "deepseek-coder:6.7b": StrategyProfile(
        name="deepseek-coder:6.7b",
        use_llm_reviewer=True,
    ),

    # ── CodeLlama family ──
    "codellama:7b": StrategyProfile(
        name="codellama:7b",
        use_llm_reviewer=True,
        import_fixes=True,           # CodeLlama frequently forgets imports
        lesson_injection=True,
    ),
    "codellama:13b": StrategyProfile(
        name="codellama:13b",
        use_llm_reviewer=True,
    ),
    "codellama:34b": StrategyProfile(
        name="codellama:34b",
        use_llm_reviewer=True,
        best_of_n=1,
    ),

    # ── StarCoder2 family ──
    "starcoder2:7b": StrategyProfile(
        name="starcoder2:7b",
        use_llm_reviewer=True,
        import_fixes=True,
        ast_fixes=True,
    ),
    "starcoder2:15b": StrategyProfile(
        name="starcoder2:15b",
        use_llm_reviewer=True,
    ),

    # ── CodeGemma family ──
    "codegemma:7b": StrategyProfile(
        name="codegemma:7b",
        use_llm_reviewer=True,
        import_fixes=True,
    ),

    # ── Llama family (general-purpose, not code-tuned) ──
    "llama3.1:8b": StrategyProfile(
        name="llama3.1:8b",
        use_llm_reviewer=True,
        import_fixes=True,           # general models forget imports more
        ast_fixes=True,
        test_calibration=True,
        lesson_injection=True,
        baseline_probes=24.5,
        benchmark_runs=2,
    ),
    "llama3.3:70b": StrategyProfile(
        name="llama3.3:70b",
        use_llm_reviewer=True,
    ),

    # ── Phi family ──
    "phi-4:14b": StrategyProfile(
        name="phi-4:14b",
        use_llm_reviewer=True,
        think_architect=False,
    ),

    # ── Codestral (Mistral) ──
    "codestral:22b": StrategyProfile(
        name="codestral:22b",
        use_llm_reviewer=True,
        best_of_n=1,
    ),

    # ── Granite Code (IBM) ──
    "granite-code:8b": StrategyProfile(
        name="granite-code:8b",
        use_llm_reviewer=True,
        import_fixes=True,
    ),
    "granite-code:20b": StrategyProfile(
        name="granite-code:20b",
        use_llm_reviewer=True,
    ),

    # ── Yi Coder ──
    "yi-coder:9b": StrategyProfile(
        name="yi-coder:9b",
        use_llm_reviewer=True,
    ),
}


def get_profile_for_model(model_name: str) -> StrategyProfile | None:
    """Find a matching profile for a given model name.

    Matching order:
    1. Exact match on key
    2. Fuzzy: profile key is a substring of model_name (longest match wins)
    """
    if model_name in STRATEGY_PROFILES:
        return STRATEGY_PROFILES[model_name]

    # Fuzzy match — longest matching key wins (avoids "7b" matching "70b")
    name_lower = model_name.lower()
    best_match: StrategyProfile | None = None
    best_len = 0
    for key, profile in STRATEGY_PROFILES.items():
        if key.lower() in name_lower and len(key) > best_len:
            best_match = profile
            best_len = len(key)

    return best_match


def load_profiles_from_json(path: Path) -> Dict[str, StrategyProfile]:
    """Load auto-generated profiles from build_profiles.py output."""
    data = json.loads(path.read_text())
    profiles: Dict[str, StrategyProfile] = {}
    for name, pdata in data.items():
        scores = [
            TechniqueScore(**s) for s in pdata.pop("technique_scores", [])
        ]
        profiles[name] = StrategyProfile(**pdata, technique_scores=scores)
    return profiles


def save_profiles_to_json(profiles: Dict[str, StrategyProfile], path: Path) -> None:
    """Save profiles to JSON for persistence across runs."""
    data = {}
    for name, profile in profiles.items():
        d = asdict(profile)
        data[name] = d
    path.write_text(json.dumps(data, indent=2))
