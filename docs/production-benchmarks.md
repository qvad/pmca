# Production Benchmark V1 — Real-World Requests

This document contains a set of high-complexity, "real-world" requests designed to push the limits of the PMCA cascade with Qwen 3.5 9B. These tasks require multi-file decomposition, error handling, and algorithmic reasoning.

---

## 1. Async Markdown Site Generator (`static_gen`)
**Category**: File I/O, Template Processing, Async
**Request**:
> Implement a `SiteGenerator` class that asynchronously converts a directory of Markdown files into HTML.
> 1. `scan(input_dir: str)`: Recursively find all `.md` files.
> 2. `parse_frontmatter(content: str) -> tuple[dict, str]`: Extract YAML front-matter (between `---` delimiters) and body.
> 3. `apply_template(body: str, meta: dict, template: str) -> str`: Replace `{{title}}`, `{{date}}`, and `{{content}}` placeholders in the template.
> 4. `build(input_dir: str, output_dir: str, template: str)`: Orchestrate the full build process using `asyncio.gather`.
>
> **Technical Constraints (CRITICAL)**:
> - **NO EXTERNAL LIBRARIES** except `PyYAML`. Do NOT use `jinja2`, `markdownify`, `mistune`, etc.
> - Use basic `re.sub()` or `str.replace()` for Markdown-to-HTML conversion (only need to handle `# Heading` -> `<h1>Heading</h1>` and paragraphs).
> - Use standard string formatting or `str.replace()` for the template system.
> - Handle `FileNotFoundError` and `yaml.YAMLError`.

## 2. Rate-Limited API Client (`api_client`)
**Category**: Networking, Concurrency, Resilience
**Request**:
> Create a robust `ResilientApiClient` for a hypothetical REST API.
> 1. `__init__(base_url: str, rate_limit: int)`: `rate_limit` is max requests per second.
> 2. `fetch_data(endpoint: str) -> dict`: Perform an async GET request with `httpx`.
> 3. Implement a `TokenBucket` or `LeakyBucket` to enforce the rate limit.
> 4. Implement exponential backoff retry logic (up to 3 retries) for 429 (Too Many Requests) and 5xx errors.
> 5. `fetch_all(endpoints: list[str]) -> list[dict]`: Fetch multiple endpoints concurrently while strictly respecting the rate limit.

## 3. Financial Portfolio Analyzer (`portfolio_pro`)
**Category**: Data Analysis, Math, CLI
**Request**:
> Implement a `PortfolioAnalyzer` that processes investment data from CSV files.
> 1. `load_csv(path: str)`: Load rows with `date`, `ticker`, `price`, `quantity`.
> 2. `calculate_returns() -> dict`: Calculate daily returns and cumulative return for each ticker.
> 3. `get_metrics() -> dict`: Compute Sharpe Ratio (assume risk-free rate 0.02), Max Drawdown, and Volatility (std dev of daily returns).
> 4. `rebalance(target_weights: dict) -> list[dict]`: Given current prices and target % weights, return a list of "buy/sell" actions to achieve the targets.
> Ensure all calculations handle `ZeroDivisionError` and missing data (NaN) by skipping those rows.

## 4. Blockchain Ledger (`tiny_chain`)
**Category**: Security, Algorithms, Integrity
**Request**:
> Create a `MinimalBlockchain` system.
> 1. `Block` class: `index`, `timestamp`, `data`, `previous_hash`, `hash`, and `nonce`.
> 2. `compute_hash(block: Block) -> str`: Use `hashlib.sha256` on the block string representation.
> 3. `mine_block(block: Block, difficulty: int)`: Implement Proof of Work — find a hash starting with `difficulty` number of zeros.
> 4. `Blockchain` class: `add_block(data: str)`, `is_chain_valid() -> bool` (re-verify all hashes and links).
> 5. Handle the genesis block automatically on initialization.

---

## Testing Strategy
For each project, the PMCA should generate:
- A clear decomposition into at least 2 files (e.g., `models.py` and `generator.py`).
- Comprehensive unit tests covering edge cases (empty directories, invalid YAML, API timeouts).
- Proper type hinting and docstrings.
