# PMCA — Portable Modular Coding Agent

A fully local coding agent with a per-model auto-tuner. Takes a vague request, produces working, tested code. Runs on a single GPU with Ollama — no cloud APIs, no per-token costs.

Tested on 5 model families. Includes PMCA-Bench: 37 tasks, 228 probes for evaluating any code generation tool.

## Quick Start

```bash
pip install -e ".[all]"
pmca setup                          # install Ollama + pull models
pmca run "Create a Calculator class with add, subtract, multiply, divide"
```

## How It Works

PMCA wraps any local LLM in a cascade: decompose → implement → repair → verify → retry. The cascade is tunable per model — the auto-tuner discovers the right configuration for any model.

```
User Request → Architect → Coder → Repair Chain → Watcher → Verified
                  │                     │              │
                  │                     │              └─ retry with lessons
                  │                     └─ 16 zero-token AST fixes
                  └─ spec with exact types, exceptions, sort directions
```

### Key Insight

Different models need different cascade depths:

| Model | Raw Score | Best PMCA Config | Finding |
|-------|-----------|-----------------|---------|
| Qwen 3.5 9B | 90% | Full cascade, max_retries=3 | Retries help — model fixes bugs from error feedback |
| DeepSeek V2 16B | 92% | Minimal — repair chain only | Retries hurt — model rewrites working code |
| Gemma 4 E2B | 85% | max_retries=0 | First answer is best — cascade should get out of the way |
| CodeGemma 7B | 70% | max_retries=0 | Same — retries degrade output |

The auto-tuner discovers these configurations automatically.

## Auto-Tuner

```bash
# Find the optimal PMCA configuration for any model
python -m pmca.tuning.cli tune \
  --model gemma4:e2b \
  --benchmark benchmark/pmca_bench.json \
  --output results/tuned_gemma4.json

# Evaluate a config against the benchmark
python -m pmca.tuning.cli evaluate \
  --config config/gemma4_e2b.yaml \
  --benchmark benchmark/pmca_bench.json
```

14 tunable parameters. Coordinate-descent optimization. Converges in 1-2 sweeps.

## PMCA-Bench

A standalone benchmark for vague-request-to-working-class code generation. 37 tasks, 228 validation probes, 3 difficulty tiers. No existing benchmark tests this task — HumanEval tests function completion, ClassEval tests skeleton filling, SWE-bench tests repo patching.

```bash
# Evaluate any tool's output against the benchmark
python benchmark/run.py --workspace ./output_dir
```

See [benchmark/README.md](benchmark/README.md) for details and published baselines.

## Model Profiler

```bash
# Profile any model's failure patterns
python scripts/profile_model.py --model deepseek-coder-v2:16b
```

Runs the model on all 37 tasks, classifies errors (import/syntax/type/logic), and recommends which repairs to enable.

## Code Quality Audit

```bash
# Run ruff + mypy + radon + spec-adherence checks on generated code
python scripts/quality_report.py
```

Published finding: Google's Gemma 4 E2B (2B effective parameters) scores 92/100 on code quality — higher than DeepSeek V2 16B (84/100). Smaller model, better spec adherence.

## Usage

### CLI

```bash
pmca run "Build a linked list with insert, delete, search"
pmca run -c config/optimal_qwen35.yaml "Implement FizzBuzz"
pmca status
pmca resume
pmca models
```

### API Server (OpenAI-compatible)

```bash
pmca serve --port 8000
```

Compatible with OpenCode, aider, Continue, Cursor — set base URL to `http://localhost:8000/v1`.

### MCP Server

```bash
pmca mcp --workspace ./workspace
```

## Configuration

YAML files in `config/`. Per-role model assignment, per-role temperature, 14 cascade parameters.

```yaml
models:
  architect:
    name: "qwen3.5-coder"
    temperature: 0.3
  coder:
    name: "qwen3.5-coder"
    temperature: 0.2
    think: false

cascade:
  max_retries: 3
  use_llm_reviewer: false
  reviewer_bypass_on_pass: true
  import_fixes: true
  ast_fixes: true
  test_calibration: true
```

Supported providers: **Ollama** (default), **Groq**, **OpenAI**, **Liquid**.

## Benchmark Results

### Raw Model Scores (228 probes, no PMCA)

| Model | Size | Probes Passed | Code Quality |
|-------|------|--------------|-------------|
| DeepSeek Coder V2 16B | 8.9 GB | 209/228 (92%) | 84/100 |
| Qwen 3.5 9B | 6.6 GB | 205/228 (90%) | — |
| Gemma 4 E2B | 6.8 GB | 193/228 (85%) | 92/100 |
| CodeGemma 7B | 5.0 GB | 160/228 (70%) | 89/100 |

### With PMCA (tuned per model)

| Model | Raw | Tuned PMCA | Key Change |
|-------|-----|-----------|------------|
| Qwen 3.5 | 90% | 100% | Full cascade |
| Gemma 4 E2B | 85% | 100%* | max_retries=0 |
| CodeGemma 7B | 70% | 50% | max_retries=0 |
| DeepSeek V2 | 92% | 75% | No tuning helped |

*On 4 calibration tasks.

### Standard Benchmarks (raw model, no PMCA)

| Model | HumanEval pass@1 | ClassEval pass@1 |
|-------|-----------------|-----------------|
| Qwen 3.5 9B | 60.4% | 30.0% |

Repair chain adds 0% on HumanEval and -1% on ClassEval. These benchmarks test code completion (fill in skeletons), not code generation from vague requests. Different task, different tool.

## Installation

Requires Python >= 3.11.

```bash
pip install -e ".[all]"        # full install
pip install -e ".[dev]"        # pytest, pytest-asyncio, pytest-mock
pip install -e ".[lint]"       # mypy, ruff
pip install -e ".[rag]"        # chromadb, sentence-transformers
```

## OpenCode / Crush Integration

Use PMCA as a backend for [OpenCode](https://github.com/opencode-ai/opencode) or [Crush](https://github.com/charmbracelet/crush) — terminal coding agents similar to Claude Code. PMCA acts as an OpenAI-compatible proxy: the agent sends coding requests, PMCA runs the cascade, and returns verified code as `write` tool calls.

```bash
# 1. Start PMCA proxy (pick a config)
OLLAMA_HOST=http://localhost:11434 pmca -c config/gemma4_32k.yaml serve --port 8222

# 2. Configure OpenCode to use PMCA
export LOCAL_ENDPOINT=http://localhost:8222/v1

# 3. Run OpenCode normally — it doesn't know PMCA is behind it
opencode
```

### Recommended Configs

| Config | Model | Speed | Best For |
|---|---|---|---|
| `gemma4_32k.yaml` | Gemma 4 E4B (8B) | ~30 tok/s | Standard tasks (96% pass rate) |
| `qwen36_8k.yaml` | Qwen 3.6 27B (GPU+CPU split) | ~4.5 tok/s | Complex tasks (100% pass rate) |
| `gemma4_quality.yaml` | Gemma 4 + quality standards | ~30 tok/s | Production-quality code (Task classes, validation) |
| `hybrid_gemma_qwen.yaml` | Gemma 4 architect + Qwen 3.5 coder | mixed | When Gemma 4 architect + Qwen implementation needed |

### How the Proxy Works

```
OpenCode → PMCA API (:8222) → cascade (architect → coder → auto-fix → verify)
                                                    ↓
                             ← tool_calls [write(file1), write(file2)]
```

Four routing modes:
1. **Tool result ack** — acknowledges tool execution, no cascade
2. **Lightweight pass-through** — title/summarizer forwarded to Ollama directly
3. **Agent mode** — coding requests run cascade, return streaming tool_calls
4. **Direct mode** — no tools in request, returns markdown with code blocks

## Language Support

PMCA detects 20 programming languages and injects idiomatic coding skills:

| Language | Skills | Test Runner |
|---|---|---|
| Python | Full (modern types, error handling, design patterns) | pytest |
| Go | Idiomatic Go, error handling, concurrency | go test |
| TypeScript | TS patterns, strict mode | jest |
| Rust | Ownership, Result/Option, derive macros | cargo test |
| Java | Records, Optional, SOLID, JUnit 5 | mvn test |
| Kotlin | Data classes, null safety, coroutines | gradle test |
| C# | Records, pattern matching, async/await | dotnet test |
| C++ | Smart pointers, RAII, string_view | ctest |
| Ruby | Duck typing, RSpec, blocks | rspec |
| PHP | Strict types, enums, PHPUnit | phpunit |
| Swift | Structs, protocols, guard clauses | XCTest |
| Scala | Case classes, pattern matching | ScalaTest |
| Elixir | Pattern matching, GenServer, pipes | ExUnit |
| JavaScript, C, Lua, R, Dart, Shell, SQL | Detection + extension mapping | varies |

### Skills Sources

Coding skills are sourced from open-source repositories:

| Source | Skills Used | License |
|---|---|---|
| [obra/superpowers](https://github.com/obra/superpowers) | systematic-debugging, verification-before-completion, code review | MIT |
| [wshobson/agents](https://github.com/wshobson/agents) | python-testing-patterns, python-error-handling, python-design-patterns, python-type-safety, python-code-style | MIT |
| [skills.sh](https://skills.sh/) | Agent Skills specification and registry | MIT |
| [vercel-labs/skills](https://github.com/vercel-labs/skills) | Skills CLI and packaging format | MIT |

Skills are injected into the coder's system prompt based on detected language. Fix skills are injected into retry prompts based on error patterns (regex, SQL, counting, assertion, import, dataclass).

## Blog Series

- [Part 1: From 36% to 100%](https://x.com/12bytez/status/2034753653786255431) — Building the repair chain, testing 6 models
- [Part 2: From 97% to Perfection](https://x.com/12bytez/status/2040288704435192031) — Qwen 3.5 thinking mode, killing the reviewer
- [Part 3: Google's Smallest Gemma 4 Just Beat a 16B Model](https://x.com/12bytez/status/2044213963857244213) — Model-agnostic tuning, code quality audit
- Part 4: The Model That Can't Use Tools Just Scored 96% — OpenCode integration, GPU+CPU split, Qwen 3.6 27B

## License

MIT

### OpenClaude Setup

[OpenClaude](https://github.com/Gitlawb/openclaude) — open-source Claude Code clone with Ollama support.

```bash
# Install
git clone https://github.com/Gitlawb/openclaude.git
cd openclaude && bun install && bun run build && npm link

# Run with local model (27B+ recommended for tool calling)
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:11434/v1
openclaude --model qwen3.6:27b --dangerously-skip-permissions

# Or via PMCA proxy (for smaller models that can't do tool calling)
export OPENAI_BASE_URL=http://localhost:8222/v1
openclaude --model pmca --dangerously-skip-permissions
```

**Model size requirements for direct tool calling (no PMCA proxy):**
- Gemma 4 E4B (8B): too small — responds conversationally
- Qwen 3.5 (9B): borderline — works in Crush, struggles in OpenClaude
- **Qwen 3.6 27B: works** — generates code via tool calls
- Models < 20B generally need PMCA proxy for agent tool calling
