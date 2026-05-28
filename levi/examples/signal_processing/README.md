# Adaptive Signal Processing Example

Implement a sliding-window denoising filter for non-stationary 1D signals.
Mirrors the setup in `benchmarks/math/signal_processing`.

Candidate function API:

```python
def run_signal_processing(noisy_signal, window_size):
    # noisy_signal: 1D numpy array of length L
    # window_size:  int (always 20 in evaluation)
    # returns: {"filtered_signal": np.ndarray of length L - window_size + 1}
```

The grader builds 5 synthetic test signals (smooth sinusoid + trend,
multi-frequency, non-stationary frequency, step changes, random walk +
trend), runs the filter on each, and combines per-signal metrics into one
score with weights `0.4 * composite + 0.2 * smoothness + 0.2 * accuracy +
0.1 * noise_reduction + 0.1 * success_rate`. Mean accuracy
(Pearson correlation with the clean signal) below 0.1 forces the score
to 0.

The grader passes ONLY the noisy signal and the window size — the clean
signal is not available to the candidate at runtime.

## Run

```bash
cd levi/examples/signal_processing
uv run python run.py --evals 20
```

Models default to BLADE's standard split (Qwen-30B mutation, GPT-5 paradigm,
OpenAI embedding). Override via `--mutation-model`, `--paradigm-model`,
`--embedding-model`, or environment variables `BLADE_MUTATION_MODEL`,
`BLADE_PARADIGM_MODEL`, `BLADE_EMBEDDING_MODEL`.
