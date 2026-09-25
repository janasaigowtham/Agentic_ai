import pytest

from trh.config import HarnessConfig, JudgeMode, ModelLineage, load_config
from trh.judge.client import MockJudgeClient, build_judge_clients
from trh.judge.markers import specialist_marker, stage_marker
from trh.judge.mock_fixtures import DEMO_TRACE_ID


def test_load_config_defaults_to_mock_with_no_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config()
    assert config.mode is JudgeMode.MOCK


def test_load_config_stays_mock_with_only_one_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config()
    assert config.mode is JudgeMode.MOCK


def test_load_config_goes_live_with_both_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "def")
    config = load_config()
    assert config.mode is JudgeMode.LIVE


def test_mock_mode_uses_claude_lineage_only_for_fact_check(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config()
    assert config.lineage_for("orient") is ModelLineage.MOCK
    assert config.lineage_for("fact_check") is ModelLineage.MOCK


def test_live_mode_lineage_split():
    config = HarnessConfig(mode=JudgeMode.LIVE, gemini_api_key="a", anthropic_api_key="b")
    assert config.lineage_for("orient") is ModelLineage.GEMINI
    assert config.lineage_for("specialist") is ModelLineage.GEMINI
    assert config.lineage_for("aggregator") is ModelLineage.GEMINI
    assert config.lineage_for("fact_check") is ModelLineage.CLAUDE


def test_mock_client_returns_specific_seeded_answer():
    client = MockJudgeClient()
    marker = specialist_marker("database", ["s004"])
    result = client.complete(stage="specialist", system_prompt="", user_prompt="", marker=marker)
    assert result["severity"] == "significant"
    assert "twice" in result["finding"]


def test_mock_client_falls_back_to_stage_default_for_unknown_marker():
    client = MockJudgeClient()
    marker = specialist_marker("database", ["s999"])
    result = client.complete(stage="specialist", system_prompt="", user_prompt="", marker=marker)
    assert result["severity"] == "none"


def test_mock_client_returns_are_independent_copies():
    client = MockJudgeClient()
    marker = specialist_marker("gate_escalation", ["s012"])
    first = client.complete(stage="specialist", system_prompt="", user_prompt="", marker=marker)
    first["severity"] = "mutated"
    second = client.complete(stage="specialist", system_prompt="", user_prompt="", marker=marker)
    assert second["severity"] == "significant"


def test_mock_client_stage_markers_for_orient_and_aggregator():
    client = MockJudgeClient()
    orient = client.complete(
        stage="orient", system_prompt="", user_prompt="", marker=stage_marker("orient", DEMO_TRACE_ID)
    )
    assert "goal" in orient and orient["goal"]

    aggregator = client.complete(
        stage="aggregator",
        system_prompt="",
        user_prompt="",
        marker=stage_marker("aggregator", DEMO_TRACE_ID),
    )
    assert len(aggregator["recommendations"]) == 3


def test_build_judge_clients_mock_mode_uses_single_client_for_every_lineage():
    config = HarnessConfig(mode=JudgeMode.MOCK, gemini_api_key=None, anthropic_api_key=None)
    clients = build_judge_clients(config)
    assert clients[ModelLineage.GEMINI] is clients[ModelLineage.CLAUDE]
    assert isinstance(clients[ModelLineage.GEMINI], MockJudgeClient)


def test_build_judge_clients_live_mode_builds_distinct_clients_without_sdk_installed():
    config = HarnessConfig(mode=JudgeMode.LIVE, gemini_api_key="a", anthropic_api_key="b")
    clients = build_judge_clients(config)
    assert clients[ModelLineage.GEMINI] is not clients[ModelLineage.CLAUDE]
    assert clients[ModelLineage.GEMINI].lineage is ModelLineage.GEMINI
    assert clients[ModelLineage.CLAUDE].lineage is ModelLineage.CLAUDE
