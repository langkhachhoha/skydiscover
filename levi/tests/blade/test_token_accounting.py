"""Token accounting for SpecEvo/BLADE runs (offline, fast).

Baselines report input/output tokens through ``cost_log.totals.json``;
SpecEvo reports them through its own ``summary.json`` / ``snapshot.json``.
These are the two links worth pinning cheaply: the client pulling ``usage``
off a completion response, and the call log accumulating it. The
whole-run wiring on top of them is covered by the orchestrator's existing
end-to-end tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from levi.blade.orchestrator import _CallLog
from levi.clients.base import ClientResult
from levi.clients.lm import _extract_reasoning_tokens, _extract_tokens


# ---------------------------------------------------------------------------
# Client-level extraction
# ---------------------------------------------------------------------------


def test_extract_tokens_openai_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45))
    assert _extract_tokens(resp) == (120, 45)


def test_extract_tokens_anthropic_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=9))
    assert _extract_tokens(resp) == (7, 9)


def test_extract_tokens_dict_usage():
    resp = SimpleNamespace(usage={"prompt_tokens": 3, "completion_tokens": 4})
    assert _extract_tokens(resp) == (3, 4)


def test_extract_tokens_missing_or_junk_is_zero():
    """Token accounting must never be able to break a search."""
    assert _extract_tokens(SimpleNamespace()) == (0, 0)
    assert _extract_tokens(SimpleNamespace(usage=None)) == (0, 0)
    assert _extract_tokens(SimpleNamespace(usage=SimpleNamespace())) == (0, 0)
    junk = SimpleNamespace(usage=SimpleNamespace(prompt_tokens="lots", completion_tokens=-5))
    assert _extract_tokens(junk) == (0, 0)


def test_client_result_defaults_to_zero_tokens():
    """CLI-backed clients report no usage; summing must still be safe."""
    r = ClientResult(text="hi", cost=0.0)
    assert (r.prompt_tokens, r.completion_tokens) == (0, 0)


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------


def test_call_log_accumulates_tokens():
    log = _CallLog()
    log.record(0.01, 100, 20)
    log.record(0.02, 50, 5)
    log.record(0.0)  # a backend that reports no usage
    assert log.calls == 3
    assert log.prompt_tokens == 150
    assert log.completion_tokens == 25
    assert log.snapshot() == {
        "llm_calls": 3,
        "cost_usd": 0.03,
        "prompt_tokens": 150,
        "completion_tokens": 25,
        "total_tokens": 175,
        "reasoning_tokens": 0,
        "latency_s": 0.0,
    }


def test_call_log_splits_by_role():
    log = _CallLog()
    log.record(0.02, 1000, 400, reasoning_tokens=300, latency_s=12.0, role="navigator")
    log.record(0.04, 1000, 600, reasoning_tokens=500, latency_s=8.0, role="navigator")
    log.record(0.001, 500, 100, latency_s=2.0, role="mutate")
    assert log.reasoning_tokens == 800
    assert log.latency_s == 22.0
    roles = log.roles_snapshot()
    assert roles["navigator"]["llm_calls"] == 2
    assert roles["navigator"]["reasoning_tokens"] == 800
    assert roles["navigator"]["mean_latency_s"] == 10.0
    assert roles["mutate"]["completion_tokens"] == 100
    assert roles["mutate"]["reasoning_tokens"] == 0


def test_extract_reasoning_tokens():
    chat = SimpleNamespace(usage=SimpleNamespace(
        completion_tokens=1120, completion_tokens_details=SimpleNamespace(reasoning_tokens=1088)))
    assert _extract_reasoning_tokens(chat) == 1088
    responses = SimpleNamespace(usage={"output_tokens_details": {"reasoning_tokens": 64}})
    assert _extract_reasoning_tokens(responses) == 64
    assert _extract_reasoning_tokens(SimpleNamespace(usage=SimpleNamespace(completion_tokens=5))) == 0
    assert _extract_reasoning_tokens(SimpleNamespace()) == 0
    junk = SimpleNamespace(usage=SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens="x")))
    assert _extract_reasoning_tokens(junk) == 0


# ---------------------------------------------------------------------------
# Role-specific clients
# ---------------------------------------------------------------------------


def _orchestrator(tmp_path, **cfg):
    from levi.blade.orchestrator import BladeConfig, BladeOrchestrator
    from levi.simple import EmbedderConfig

    return BladeOrchestrator(BladeConfig(
        problem_description="p", function_signature="def solve(x):", score_fn=lambda f: {"score": 0.0},
        fn_name="solve", output_dir=tmp_path, embedder_config=EmbedderConfig(model="fake/embed", dim=8), **cfg,
    ))


def test_navigator_and_advisor_default_to_their_roles_clients(tmp_path):
    orch = _orchestrator(tmp_path)
    assert orch.navigator_lm is orch.paradigm_lm
    assert orch.advisor_lm is orch.mutation_lm
    # Resolved at call time: swapping the role's client reaches them too.
    fake = SimpleNamespace(model="fake/paradigm")
    orch.paradigm_lm = fake
    assert orch.navigator_lm is fake


def test_navigator_and_advisor_overrides(tmp_path):
    orch = _orchestrator(
        tmp_path,
        navigator_model="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        advisor_model="openrouter/openai/gpt-5",
    )
    assert orch.navigator_lm is not orch.paradigm_lm
    assert orch.navigator_lm.model == "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
    assert orch.advisor_lm.model == "openrouter/openai/gpt-5"
    # The bootstrap's diverse seeds keep the paradigm model.
    assert orch.paradigm_lm.model == "openrouter/openai/gpt-5"


def test_navigator_reasoning_effort_alone_gets_its_own_client(tmp_path):
    orch = _orchestrator(tmp_path, navigator_reasoning_effort="low")
    assert orch.navigator_lm is not orch.paradigm_lm
    assert orch.navigator_lm.model == orch.paradigm_lm.model
    assert orch.navigator_lm.defaults == {"reasoning_effort": "low"}
    assert orch.paradigm_lm.defaults == {}


def test_call_log_snapshot_is_a_frozen_copy():
    """This is what keeps the init-phase split from tracking the search."""
    log = _CallLog()
    log.record(0.01, 10, 2)
    snap = log.snapshot()
    log.record(0.01, 10, 2)
    assert snap["prompt_tokens"] == 10
