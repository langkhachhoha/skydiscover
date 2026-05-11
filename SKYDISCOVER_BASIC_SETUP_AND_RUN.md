# SkyDiscover Basic Setup And Run

This is a short operational guide for local runs and GitHub Actions smoke tests.

## Requirements

- Python 3.10 or newer.
- `uv`.
- An LLM API key.
- Docker only if you pass a Docker evaluator directory. Native Python evaluators do not need Docker.

Install base dependencies:

```bash
uv sync
```

For math benchmarks with heavier dependencies:

```bash
uv sync --extra math
```

The `circle_packing` smoke test only needs the base environment plus `numpy`, so `uv sync` is enough.

## API Keys

Local `.env` example:

```bash
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

SkyDiscover can run OpenRouter GPT-5 with:

```bash
--model openrouter/openai/gpt-5
```

For GitHub Actions, store the key in:

```text
Settings -> Secrets and variables -> Actions -> New repository secret
Name: OPENROUTER_API_KEY
Value: <your OpenRouter key>
```

Do not commit `.env`.

## Run Circle Packing Locally

EvoX, 2 iterations:

```bash
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search evox \
  --model openrouter/openai/gpt-5 \
  --iterations 2 \
  --output outputs/local/circle_packing_evox_2
```

AdaEvolve, 2 iterations:

```bash
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search adaevolve \
  --model openrouter/openai/gpt-5 \
  --iterations 2 \
  --output outputs/local/circle_packing_adaevolve_2
```

## Native Evaluator Versus Docker Evaluator

Native Python evaluator:

```bash
... benchmarks/math/circle_packing/evaluator.py ...
```

Docker evaluator:

```bash
... benchmarks/math/circle_packing/evaluator/ ...
```

The difference matters. If the evaluator argument is a directory with `Dockerfile` and `evaluate.sh`, SkyDiscover uses Docker. If it is a Python file with `evaluate(program_path)`, SkyDiscover runs it in the host Python environment.

## GitHub Actions Smoke Test

This repo includes:

```text
.github/workflows/circle-packing-evox.yml
```

Run it manually:

1. Push the workflow file to GitHub.
2. Add the `OPENROUTER_API_KEY` repository secret.
3. Open the GitHub repo.
4. Go to `Actions`.
5. Select `Circle Packing EvoX Smoke`.
6. Click `Run workflow`.
7. Keep `iterations=2` for the first test.

The workflow uploads run outputs as an artifact named:

```text
circle-packing-evox-output-<run_id>
```

## Outputs

Typical output structure:

```text
outputs/<run-name>/
  logs/
  best/
    best_program.py
    best_program_info.json
  checkpoints/
    checkpoint_*/
```

Important files:

- `best/best_program.py`: current best generated solution.
- `best/best_program_info.json`: best metrics.
- `logs/*.log`: detailed run logs.
- `checkpoints/checkpoint_<n>`: resume point.

Resume from checkpoint:

```bash
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search evox \
  --model openrouter/openai/gpt-5 \
  --iterations 10 \
  --checkpoint outputs/local/circle_packing_evox_2/checkpoints/checkpoint_2
```

## Quick Troubleshooting

Missing key:

```text
Missing GitHub secret: OPENROUTER_API_KEY
```

Add the repository secret or export the key locally.

Docker not found:

```text
docker: command not found
```

Use a Python evaluator file instead of an evaluator directory, or install Docker.

Model/API errors:

```bash
uv run python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
resp = client.chat.completions.create(
    model="openai/gpt-5",
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    max_completion_tokens=128,
)
print(resp.choices[0].message.content)
PY
```

Dependency errors:

```bash
uv sync
uv sync --extra math
```

Use benchmark-specific `requirements.txt` only when that benchmark needs it.
