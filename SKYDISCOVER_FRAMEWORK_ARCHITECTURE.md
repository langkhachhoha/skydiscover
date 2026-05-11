# SkyDiscover Framework Architecture

This document is a developer map for agents or engineers extending SkyDiscover. It focuses on how the framework is wired, where responsibilities live, and which extension points should be reused before adding new abstractions.

## Top-Level Flow

SkyDiscover runs an optimization loop around a candidate program:

1. Load a `Config` from YAML and runtime CLI/API overrides.
2. Load an initial program, or start from scratch if no seed is provided.
3. Create a search database and discovery controller for the selected search algorithm.
4. Build an LLM prompt from the current database state and benchmark context.
5. Call an OpenAI-compatible LLM backend.
6. Parse the model output into a candidate solution.
7. Evaluate the candidate with a Python, Docker, or Harbor evaluator.
8. Store metrics, artifacts, checkpoints, and the best program.

The main runtime path is:

```text
skydiscover/cli.py
  -> skydiscover.config.load_config/apply_overrides
  -> skydiscover.runner.Runner
  -> skydiscover.search.registry.create_database
  -> skydiscover.search.route.get_discovery_controller
  -> skydiscover.search.default_discovery_controller.DiscoveryController
  -> skydiscover.evaluation.create_evaluator
  -> skydiscover.llm.llm_pool.LLMPool
```

## CLI And Public API

`skydiscover/cli.py` defines the `skydiscover-run` command. It parses the initial program path, evaluator path, config path, model, API base, search algorithm, output directory, and iteration count.

The CLI is intentionally thin. It should not contain search logic. It loads config, applies overrides, handles external backend routing, creates a `Runner`, and prints the final result.

`skydiscover/api.py` exposes Python API helpers such as `run_discovery`. Use this layer when adding programmatic entry points rather than duplicating runner setup.

## Configuration

`skydiscover/config.py` is the central configuration schema. Important dataclasses:

- `Config`: top-level run configuration.
- `LLMConfig` and `LLMModelConfig`: provider, model, API key, generation parameters, retry and timeout settings.
- `SearchConfig`: search type, search database config, context program count, EvoX-specific sharing and switch settings.
- `EvaluatorConfig`: evaluator path, timeout, cascade settings, LLM-as-judge switch.
- `DatabaseConfig` subclasses: algorithm-specific database parameters.
- `MonitorConfig`: live dashboard settings.

Provider/model parsing is handled by `_parse_model_spec`. Provider prefixes like `gemini/...`, `anthropic/...`, and `openrouter/...` resolve base URLs and environment variable names. Runtime overrides are applied by `apply_overrides`.

When adding provider support, update `_PROVIDERS`, check key resolution, and verify model names are passed in the exact format the provider expects.

## Runner

`skydiscover/runner.py` owns end-to-end run orchestration:

- Creates the output directory.
- Loads the initial program.
- Creates the search database.
- Creates the discovery controller.
- Adds the initial program to the database.
- Starts/stops the monitor.
- Saves checkpoints and best program artifacts.
- Re-evaluates the best program in test mode at the end.

The runner is the right place for run lifecycle behavior. It is not the right place for algorithm-specific selection, prompt construction, or evaluation scoring.

## Search Registry And Routing

Search components are registered in `skydiscover/search/route.py` and instantiated through `skydiscover/search/registry.py`.

Key registry concepts:

- `register_database(search_type, database_class)`
- `register_controller(search_type, controller_class)`
- `create_database(search_type, config)`
- `get_program(config, initial_solution, id, metrics, iteration)`

Most algorithms only need a database implementation. Algorithms with custom orchestration provide a controller. For example, AdaEvolve and EvoX both register custom controllers.

## Default Discovery Controller

`skydiscover/search/default_discovery_controller.py` implements the normal generate-evaluate loop:

1. Sample parent/context programs from the database.
2. Build prompts with a context builder.
3. Call the LLM or agentic generator.
4. Parse diffs or full rewrites.
5. Evaluate the candidate.
6. Postprocess the result into the database.

This class contains reusable primitives for search algorithms. If a new algorithm still follows the same generate/evaluate/postprocess pattern, prefer changing the database sampling policy rather than creating a new controller.

## Search Databases

Databases store `Program` records and define sampling/selection behavior. Shared base types live in `skydiscover/search/base_database.py`.

Existing database implementations:

- `skydiscover/search/topk`: keeps and samples high-performing programs.
- `skydiscover/search/best_of_n`: generates several variants and keeps the best.
- `skydiscover/search/beam_search`: beam-style program selection.
- `skydiscover/search/adaevolve`: adaptive multi-island archive and policy logic.
- `skydiscover/search/openevolve_native`: native OpenEvolve-like database.
- `skydiscover/search/gepa_native`: GEPA-style candidate pool and acceptance logic.
- `skydiscover/search/evox/database`: EvoX search strategy database.

Database extensions should preserve the `ProgramDatabase` contract: add programs, sample context, compute best program, expose statistics, and support serialization for checkpoints.

## EvoX

EvoX lives under `skydiscover/search/evox/`.

Main files:

- `controller.py`: `CoEvolutionController`, the main EvoX orchestration.
- `database/initial_search_strategy.py`: initial solution-search strategy used by EvoX.
- `database/search_strategy_db.py`: database for evolved search strategies.
- `database/search_strategy_evaluator.py`: evaluator for search strategies.
- `config/search.yaml`: config used for meta-evolution of search strategies.
- `utils/variation_operator_generator.py`: generates/refines variation operator labels.
- `utils/search_scorer.py`: scores search strategy windows.
- `utils/coevolve_logging.py`: saves EvoX-specific logs and artifacts.

EvoX has two nested optimization processes:

- Solution evolution: improves benchmark solutions.
- Search-strategy evolution: improves the algorithm used to sample and guide solution evolution.

`CoEvolutionController` subclasses the default discovery controller but interleaves normal solution iterations with search-strategy evolution. It can inherit the parent LLM config into the search-level controller through `search.share_llm`.

## Prompt And Context Builders

Prompt construction lives in `skydiscover/context_builder/`.

Important builders:

- `default`: standard program optimization prompts.
- `adaevolve`: AdaEvolve-specific prompt templates.
- `evox`: EvoX-specific summaries, strategy evolution prompts, and formatters.
- `gepa_native`: GEPA-specific rewrite/diff prompts.

Templates are plain text files under each builder's `templates/` directory. Prefer editing templates or builder formatting helpers over hardcoding long prompts in controllers.

## LLM Layer

LLM access is under `skydiscover/llm/`.

- `base.py`: common response/interface types.
- `openai.py`: OpenAI-compatible Chat Completions and Responses API backend.
- `llm_pool.py`: weighted model selection across configured models.
- `agentic_generator.py`: multi-step tool-calling generator for agentic mode.
- `responses_utils.py`: helpers for Responses API output parsing.

The framework expects OpenAI-compatible providers where possible. Provider-specific logic should stay in config parsing or the LLM backend, not in search controllers.

## Evaluators

Evaluator dispatch lives in `skydiscover/evaluation/__init__.py`.

Detection order:

1. Harbor task: directory with `instruction.md`, `tests/test.sh`, and `environment/Dockerfile`.
2. Containerized evaluator: directory with `Dockerfile` and `evaluate.sh`.
3. Python evaluator: file with `evaluate(program_path)`.

Evaluator implementations:

- `evaluator.py`: native Python evaluator.
- `container_evaluator.py`: Docker-based persistent container evaluator.
- `harbor_evaluator.py`: Harbor benchmark protocol.
- `llm_judge.py`: optional LLM-as-judge scorer.
- `evaluation_result.py`: normalized metrics/artifacts result.

For CI without Docker, pass a Python evaluator file directly. Passing a directory containing `Dockerfile` and `evaluate.sh` will invoke Docker.

Bundled Docker evaluator directories usually contain a `requirements.txt` next to `evaluator.py`. For native runs, install those dependencies into the active environment with `scripts/install_benchmark_requirements.py` and pass the Python evaluator file instead of the evaluator directory. This preserves the dependency setup without using Docker for benchmarks whose Dockerfile only performs Python package installation.

## Benchmarks

Benchmarks live in `benchmarks/`. A normal benchmark includes:

- `initial_program.py` or `initial_program.cpp`, optionally with `EVOLVE-BLOCK` markers.
- `config.yaml`.
- `evaluator.py` for native evaluation, or `evaluator/` for Docker evaluation.
- Optional `requirements.txt`, data download scripts, references, and README files.

Some benchmark families have special runtime needs:

- `math`: mostly Python numerical optimization.
- `ADRS`: systems benchmarks; many provide Docker evaluator directories.
- `gpu_mode`: native Python evaluator, but requires CUDA GPU or Modal.
- `ale_bench`: native Python evaluator around ALE Bench and C++ solutions.
- `frontier-cs-eval`: Python file exists, but internally uses Docker judge backend.
- `image_gen`: image generation and LLM-as-judge.
- `prompt_optimization`: DSPy and dataset-heavy prompt optimization.
- `kernelbench`: supports Docker and native modes via resolver config.

## Outputs And Checkpoints

Run outputs are written under `outputs/` by default, or the directory passed through `--output`.

Common output files:

- `logs/*.log`: run logs.
- `best/best_program.py`: best solution.
- `best/best_program_info.json`: metrics and metadata.
- `checkpoints/checkpoint_<n>/`: serialized database and best program state.
- `programs/*.json`: program-level records in checkpoints.

Use checkpoint directories with `--checkpoint` to resume.

## Extension Guidelines

Add a new benchmark by creating a config, seed program, and evaluator. Use a Python evaluator first unless isolation or system dependencies require Docker.

Add a new search algorithm by first deciding whether a database-only implementation is enough. If the standard generate/evaluate loop works, implement a database and register it. Add a controller only when orchestration differs materially.

Add a new provider by extending config provider resolution and, only if required, the OpenAI-compatible LLM backend.

Keep evaluator-specific dependencies inside benchmark extras or benchmark `requirements.txt`. Keep search algorithms independent of benchmark-specific imports.
