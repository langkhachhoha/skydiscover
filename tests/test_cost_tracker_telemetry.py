"""Per-call telemetry in the cost log, and the runner's wall-clock budget."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from skydiscover.llm import cost_tracker
from skydiscover.runner import Runner


def _response(**usage):
    return SimpleNamespace(id="gen-1", usage=SimpleNamespace(**usage))


def test_usage_record_reads_reasoning_tokens_from_either_api():
    chat = _response(prompt_tokens=10, completion_tokens=1120,
                     completion_tokens_details=SimpleNamespace(reasoning_tokens=1088))
    assert cost_tracker._extract_usage_record(chat)["reasoning_tokens"] == 1088
    responses = _response(input_tokens=10, output_tokens=70,
                          output_tokens_details={"reasoning_tokens": 64})
    assert cost_tracker._extract_usage_record(responses)["reasoning_tokens"] == 64
    plain = _response(prompt_tokens=10, completion_tokens=5)
    assert cost_tracker._extract_usage_record(plain)["reasoning_tokens"] is None


def test_record_call_logs_latency_and_reasoning(tmp_path, monkeypatch):
    log = tmp_path / "cost_log.jsonl"
    monkeypatch.setenv("SKYDISCOVER_COST_LOG", str(log))
    monkeypatch.delenv("SKYDISCOVER_MAX_COST_USD", raising=False)
    for name in ("_TOTAL_COST", "_TOTAL_CALLS", "_TOTAL_PROMPT_TOKENS", "_TOTAL_COMPLETION_TOKENS",
                 "_TOTAL_REASONING_TOKENS", "_TOTAL_LATENCY_S"):
        monkeypatch.setattr(cost_tracker, name, 0)

    for latency, reasoning in ((12.5, 900), (7.5, 100)):
        cost_tracker.record_call(
            _response(prompt_tokens=100, completion_tokens=1000, cost=0.01,
                      completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)),
            model="openai/gpt-5", api_base="https://openrouter.ai/api/v1",
            api_kind="chat.completions", call_site="test", latency_s=latency,
        )

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["latency_s"] for r in rows] == [12.5, 7.5]
    assert [r["reasoning_tokens"] for r in rows] == [900, 100]
    totals = json.loads(log.with_suffix(".totals.json").read_text())
    assert totals["total_reasoning_tokens"] == 1000
    assert totals["total_latency_s"] == 20.0


def _fake_runner():
    controller = SimpleNamespace(early_stopping_triggered=False, stops=0)
    controller.request_shutdown = lambda: setattr(controller, "stops", controller.stops + 1)
    return SimpleNamespace(discovery_controller=controller), controller


def test_time_budget_requests_a_graceful_stop(monkeypatch):
    monkeypatch.setenv("SKYDISCOVER_MAX_SECONDS", "0.05")
    runner, controller = _fake_runner()
    Runner._start_time_budget(runner)
    deadline = time.time() + 5
    while controller.stops == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert controller.stops == 1
    assert controller.early_stopping_triggered is True


def test_no_time_budget_when_unset_or_blank(monkeypatch):
    for value in (None, "", "0", "junk"):
        if value is None:
            monkeypatch.delenv("SKYDISCOVER_MAX_SECONDS", raising=False)
        else:
            monkeypatch.setenv("SKYDISCOVER_MAX_SECONDS", value)
        runner, controller = _fake_runner()
        Runner._start_time_budget(runner)
        time.sleep(0.05)
        assert controller.stops == 0
